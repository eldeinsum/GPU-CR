pub mod backend;
pub mod comm;
pub mod constants;
pub mod copy;
pub mod cuda;
pub mod layout;
pub mod runtime;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    CString(std::ffi::NulError),
    Cuda { code: i32, message: String },
    Hip { code: i32, message: String },
    Protocol(String),
    Process(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(err) => write!(f, "io error: {err}"),
            Error::CString(err) => write!(f, "invalid c string: {err}"),
            Error::Cuda { code, message } => write!(f, "cuda error {code}: {message}"),
            Error::Hip { code, message } => write!(f, "hip error {code}: {message}"),
            Error::Protocol(message) => write!(f, "protocol error: {message}"),
            Error::Process(message) => write!(f, "process error: {message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value)
    }
}

impl From<std::ffi::NulError> for Error {
    fn from(value: std::ffi::NulError) -> Self {
        Error::CString(value)
    }
}
