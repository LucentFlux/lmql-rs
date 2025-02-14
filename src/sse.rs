//! LLM streaming uses SSE (Server-Sent Events) to stream responses from the server to the client.
//! This module provides a client for SSE built on top of Hyper.

use std::io::{BufRead, Read};
use std::sync::Arc;

use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use rustls_pki_types::ServerName;
use tokio::select;
use tokio::{
    net::TcpStream,
    sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
};
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

const TIMEOUT_MS: u64 = 10000;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Io error")]
    IoError(#[from] std::io::Error),
    #[error("Hyper error")]
    HyperError(#[from] hyper::Error),
    #[error("Http error")]
    HttpError(#[from] hyper::http::Error),
    #[error("Json error")]
    JsonError(#[from] serde_json::Error),
}

type Result<T> = std::result::Result<T, Error>;

pub(crate) struct SseClient {
    _join_handle: tokio::task::JoinHandle<()>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    rx: UnboundedReceiver<Result<SseValue>>,
}

#[derive(Debug)]
pub(crate) struct SseValue {
    pub(crate) event: String,
    pub(crate) value: serde_json::Value,
}

async fn receive_events(
    mut res: Response<Incoming>,
    tx: UnboundedSender<Result<SseValue>>,
) -> Result<()> {
    let mut accumulation = Vec::new();

    while let Some(next) = res.frame().await {
        let frame = next?;
        if let Some(chunk) = frame.data_ref() {
            let mut chunk = &**chunk;

            // We split on double newlines, respecting the accumulation buffer.
            let mut i = 0;
            while !chunk.is_empty() && i < chunk.len() - 1 {
                if chunk[i] == b'\n' && chunk[i + 1] == b'\n' {
                    let (message_end, tail) = chunk.split_at(i);
                    chunk = &tail[2..];
                    i = 0;

                    let mut message = std::io::Read::chain(
                        std::io::Cursor::new(&accumulation),
                        std::io::Cursor::new(message_end),
                    );

                    let mut staging = String::new();
                    let mut data = String::new();
                    let mut event = String::new();
                    loop {
                        let mut header = [0u8; 4];
                        if message.read_exact(&mut header).is_err() {
                            break;
                        }

                        match &header {
                            b"data" => {
                                // Last 2 bytes
                                let mut header_colon = [0u8; 2];
                                message.read_exact(&mut header_colon)?;
                                assert_eq!(&header_colon, b": ");

                                message.read_line(&mut data)?;
                                if data.ends_with('\n') {
                                    data.pop(); // Remove the trailing newline.
                                }
                            }
                            b"even" => {
                                // Last 3 bytes
                                let mut header_colon = [0u8; 3];
                                message.read_exact(&mut header_colon)?;
                                assert_eq!(&header_colon, b"t: ");

                                message.read_line(&mut event)?;
                                if event.ends_with('\n') {
                                    event.pop(); // Remove the trailing newline.
                                }
                            }
                            _ => {
                                message.read_line(&mut staging)?;
                            }
                        }
                    }

                    let value = serde_json::from_str(&data)?;
                    if let Err(_) = tx.send(Ok(SseValue { event, value })) {
                        tracing::error!("stream disconnected prematurely");
                        return Ok(());
                    }

                    accumulation.clear();
                } else {
                    i += 1;
                }
            }
            accumulation.extend_from_slice(chunk);
        }
    }

    Ok(())
}

async fn run_client(
    request: Request<String>,
    tx: UnboundedSender<Result<SseValue>>,
    shutdown_signal: tokio::sync::oneshot::Receiver<()>,
) -> Result<()> {
    let url = request.uri();

    let host = url.host().expect("Url should have a host");
    let port = url.port_u16().unwrap_or(443);

    let mut root_cert_store = RootCertStore::empty();
    root_cert_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let mut config = ClientConfig::builder()
        .with_root_certificates(root_cert_store)
        .with_no_client_auth();
    config.alpn_protocols = vec![b"h2".to_vec()];
    let connector = TlsConnector::from(Arc::new(config));

    let tls_domain = ServerName::try_from(host.to_string())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid dnsname"))?;

    let stream = TcpStream::connect(format!("{}:{}", host, port)).await?;
    let stream = connector.connect(tls_domain, stream).await?;

    let executor = hyper_util::rt::tokio::TokioExecutor::new();
    let io = TokioIo::new(stream);
    let (mut sender, connection) = hyper::client::conn::http2::handshake(executor, io).await?;

    tokio::task::spawn(async move {
        if let Err(e) = connection.await {
            tracing::error!("connection error: {}", e);
        }
        tracing::debug!("connection closed");
    });

    let work = sender.send_request(request);
    let res = match tokio::time::timeout(std::time::Duration::from_millis(TIMEOUT_MS), work).await {
        Ok(result) => result?,
        Err(_) => {
            return Err(tokio::io::Error::new(tokio::io::ErrorKind::TimedOut, "Timeout").into())
        }
    };

    if !res.status().is_success() {
        return Err(tokio::io::Error::new(
            tokio::io::ErrorKind::Other,
            format!("request failed with status: {}", res.status()),
        )
        .into());
    }

    select! {
        _ = receive_events(res, tx) => {
            // Connection was probably closed
        }
        _ = shutdown_signal => {
            // Received a shutdown signal
        }
    };
    Ok(())
}

impl SseClient {
    pub(crate) fn spawn(request: Request<String>) -> Self {
        let (tx, rx) = unbounded_channel();
        let (shutdown, shutdown_signal) = tokio::sync::oneshot::channel::<()>();

        let join_handle = tokio::spawn(async move {
            let tx_clone = tx.clone();
            if let Err(e) = run_client(request, tx_clone, shutdown_signal).await {
                let _ = tx.send(Err(e));
            }
        });

        Self {
            _join_handle: join_handle,
            rx,
            shutdown: Some(shutdown),
        }
    }
}

impl futures::Stream for SseClient {
    type Item = Result<SseValue>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for SseClient {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            if !shutdown.is_closed() {
                shutdown.send(()).ok();
            }
        }
    }
}
