def print_missing_tools_warning(missing: list) -> None:
    """Prints the same warning block `check_dependencies` used to print inline."""
    if not missing:
        return
    print("WARNING: The following required tools were not found in your PATH:")
    for tool in missing:
        print(f"  - {tool}")
    print("Please resolve these dependencies for full functionality.")
    print("-" * 50)
