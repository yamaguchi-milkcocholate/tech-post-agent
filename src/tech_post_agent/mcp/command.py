def filesystem_mcp(name: str, fs_dir: str) -> dict:
    return {
        name: {
            "command": "docker",
            "args": [
                "run",
                "-i",
                "--rm",
                "-v",
                f"{fs_dir}:/data:ro",  # 読み取り専用でマウント
                "--user",
                "1000:1000",
                "-e",
                "FS_ROOT=/data",
                "-e",
                "FS_MODE=read",
                "mcp/filesystem",
                "/data",  # filesystemサーバーに許可するディレクトリを指定
            ],
            "transport": "stdio",
        }
    }
