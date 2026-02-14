def format_seconds(value: float) -> str:
    if value <= 0:
        return "0.0 s"
    if value < 60:
        return f"{value:.1f} s"
    minutes = int(value // 60)
    seconds = value % 60
    return f"{minutes}m {seconds:.1f}s"

