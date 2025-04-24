# roop/status_utils.py
def update_status(message: str, source: str = "ROOP.CORE") -> None:
    print(f"[{source}] {message}")
