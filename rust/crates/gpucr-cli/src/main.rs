use std::env;
use std::process;

use gpucr::backend::{default_checkpoint_path_for_pid, ensure_checkpoint_dir};
use gpucr::comm::Comm;
use gpucr::constants::signals;
use gpucr::constants::{CKPT_MSG, INIT_MSG, RESTORE_MSG};
use gpucr::runtime::run_cuda_checkpoint_toggle;
use gpucr::{Error, Result};

enum Command {
    Init {
        pid: libc::pid_t,
        path: String,
    },
    Checkpoint {
        pid: libc::pid_t,
        path: String,
        buffer_only: bool,
        init_first: bool,
    },
    Restore {
        pid: libc::pid_t,
        buffer_only: bool,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("gpucr-client: {err}");
        process::exit(1);
    }
}

fn run() -> Result<()> {
    ensure_checkpoint_dir()?;
    match parse_args()? {
        Command::Init { pid, path } => init(pid, &path),
        Command::Checkpoint {
            pid,
            path,
            buffer_only,
            init_first,
        } => checkpoint(pid, &path, buffer_only, init_first),
        Command::Restore { pid, buffer_only } => restore(pid, buffer_only),
    }
}

fn parse_args() -> Result<Command> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        return usage_error(&args[0]);
    }
    if args[1].starts_with('-') {
        return parse_legacy_args(&args);
    }
    let command = args[1].as_str();
    let pid = args[2]
        .parse::<libc::pid_t>()
        .map_err(|err| Error::Protocol(format!("invalid pid '{}': {err}", args[2])))?;
    match command {
        "init" => {
            let path = args
                .get(3)
                .cloned()
                .unwrap_or_else(|| default_checkpoint_path_for_pid(pid));
            Ok(Command::Init { pid, path })
        }
        "checkpoint" | "ckpt" => {
            let path = args
                .get(3)
                .cloned()
                .unwrap_or_else(|| default_checkpoint_path_for_pid(pid));
            Ok(Command::Checkpoint {
                pid,
                path,
                buffer_only: false,
                init_first: true,
            })
        }
        "restore" => Ok(Command::Restore {
            pid,
            buffer_only: false,
        }),
        _ => usage_error(&args[0]),
    }
}

fn parse_legacy_args(args: &[String]) -> Result<Command> {
    let mut init = false;
    let mut checkpoint = false;
    let mut restore = false;
    let mut buffer_only = false;
    let mut pid = None;
    let mut path = None;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "-i" => init = true,
            "-c" => checkpoint = true,
            "-r" => restore = true,
            "-b" => buffer_only = true,
            "-p" => {
                idx += 1;
                let value = args
                    .get(idx)
                    .ok_or_else(|| Error::Protocol("missing pid after -p".to_string()))?;
                pid = Some(
                    value
                        .parse::<libc::pid_t>()
                        .map_err(|err| Error::Protocol(format!("invalid pid '{value}': {err}")))?,
                );
            }
            "-m" => {
                idx += 1;
                if args.get(idx).is_none() {
                    return Err(Error::Protocol("missing pid after -m".to_string()));
                }
            }
            "--path" | "-o" => {
                idx += 1;
                path = Some(
                    args.get(idx)
                        .ok_or_else(|| Error::Protocol("missing path".to_string()))?
                        .clone(),
                );
            }
            other => {
                return Err(Error::Protocol(format!("unknown argument '{other}'")));
            }
        }
        idx += 1;
    }

    let mode_count = init as u8 + checkpoint as u8 + restore as u8;
    if mode_count != 1 {
        return usage_error(&args[0]);
    }
    let pid = pid.ok_or_else(|| Error::Protocol("missing required -p pid".to_string()))?;
    let path = path.unwrap_or_else(|| default_checkpoint_path_for_pid(pid));
    if init {
        Ok(Command::Init { pid, path })
    } else if checkpoint {
        Ok(Command::Checkpoint {
            pid,
            path,
            buffer_only,
            init_first: false,
        })
    } else {
        Ok(Command::Restore { pid, buffer_only })
    }
}

fn usage_error(binary: &str) -> Result<Command> {
    Err(Error::Protocol(format!(
        "usage: {binary} <init|checkpoint|restore> <pid> [checkpoint_path] or {binary} [-i|-c|-r] -p <pid> [-b]"
    )))
}

fn init(pid: libc::pid_t, path: &str) -> Result<()> {
    let mut comm = Comm::for_pid(pid)?;
    comm.controls_mut().set_checkpoint_path(path);
    comm.send_msg(INIT_MSG);
    signal(pid, signals::cr_init())?;
    comm.wait_for_finish()?;
    Ok(())
}

fn checkpoint(pid: libc::pid_t, path: &str, buffer_only: bool, init_first: bool) -> Result<()> {
    if init_first {
        init(pid, path)?;
    }
    let comm = Comm::for_pid(pid)?;
    comm.send_msg(CKPT_MSG);
    signal(pid, signals::cr_checkpoint())?;
    comm.wait_for_finish()?;

    if !buffer_only {
        run_cuda_checkpoint_toggle(pid)?;
    }
    Ok(())
}

fn restore(pid: libc::pid_t, buffer_only: bool) -> Result<()> {
    let comm = Comm::for_pid(pid)?;
    if !buffer_only {
        run_cuda_checkpoint_toggle(pid)?;
    }
    comm.send_msg(RESTORE_MSG);
    signal(pid, signals::cr_restore())?;
    comm.wait_for_finish()?;
    Ok(())
}

fn signal(pid: libc::pid_t, signal: i32) -> Result<()> {
    let rc = unsafe { libc::kill(pid, signal) };
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::Io(std::io::Error::last_os_error()))
    }
}
