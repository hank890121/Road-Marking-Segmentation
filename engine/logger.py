import logging
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir: str) -> None:
        FORMAT = "%(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=FORMAT,
            datefmt="[%X]",
            handlers=[RichHandler()],
        )
        self.logger = logging.getLogger("rich")
        self.writer = SummaryWriter(logdir)

    def debug(self, title: str, msg: str) -> None:
        self.logger.debug(f"[{title}] {msg}", stacklevel=2)

    def info(self, title: str, msg: str) -> None:
        self.logger.info(f"[{title}] {msg}", stacklevel=2)

    def warn(self, title: str, msg: str) -> None:
        self.logger.warn(f"[{title}] {msg}", stacklevel=2)

    def error(self, title: str, msg: str) -> None:
        self.logger.error(f"[{title}] {msg}", stacklevel=2)

    def tb_log(self, title: str, value: float, iteration: int) -> None:
        self.writer.add_scalar(title, value, iteration)
