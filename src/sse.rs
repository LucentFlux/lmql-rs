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

pub(crate) struct SseClient {
    _join_handle: tokio::task::JoinHandle<()>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    rx: UnboundedReceiver<Event<SseValue>>,
}

#[derive(Debug)]
pub(crate) enum Event<T> {
    Failed(Error),
    Message(T),
    Shutdown,
}

#[derive(Debug)]
pub(crate) struct SseValue {
    pub(crate) event: String,
    pub(crate) value: serde_json::Value,
}

async fn receive_events(
    mut res: Response<Incoming>,
    tx: UnboundedSender<Event<SseValue>>,
) -> Result<(), Error> {
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

                    let mut event_header = [0u8; 7];
                    message.read(&mut event_header)?;
                    assert_eq!(&event_header, b"event: ");

                    let mut event = String::new();
                    message.read_line(&mut event)?;
                    event.pop(); // Remove the trailing newline.

                    let mut data_header = [0u8; 6];
                    message.read(&mut data_header)?;
                    assert_eq!(&data_header, b"data: ");

                    let value = serde_json::from_reader(message)?;
                    if let Err(_) = tx.send(Event::Message(SseValue { event, value })) {
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
    tx: UnboundedSender<Event<SseValue>>,
    shutdown_signal: tokio::sync::oneshot::Receiver<()>,
) -> Result<(), Error> {
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
                let _ = tx.send(Event::Failed(e));
            } else {
                let _ = tx.send(Event::Shutdown);
            }
        });

        Self {
            _join_handle: join_handle,
            rx,
            shutdown: Some(shutdown),
        }
    }

    pub(crate) async fn next(&mut self) -> Event<SseValue> {
        self.rx.recv().await.unwrap_or(Event::Shutdown)
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
