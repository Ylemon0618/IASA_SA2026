import re
import sys

import requests


def get_google_drive_file_id(url):
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if match:
        return match.group(1)

    if re.match(r"^[a-zA-Z0-9_-]+$", url):
        return url

    return None


def download_from_google_drive(drive_url, output_filename):
    file_id = get_google_drive_file_id(drive_url)

    if not file_id:
        print("[-] 유효한 구글 드라이브 링크나 파일 ID를 찾을 수 없습니다.")
        return

    print(f"[*] Original Link : {drive_url}")
    print(f"[*] File ID       : {file_id}")
    print(f"[*] Saving to     : {output_filename}")

    download_url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
        }
    )

    try:
        print("[*] Starting download...")
        response = session.get(
            download_url, params={"id": file_id}, stream=True
        )

        # 대용량 파일 경고(Confirm Token) 확인
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                params = {"id": file_id, "confirm": value}
                response = session.get(
                    download_url, params=params, stream=True
                )
                break

        total_size = int(response.headers.get("content-length", 0))
        read_so_far = 0
        chunk_size = 32768

        with open(output_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    read_so_far += len(chunk)

                    if total_size > 0:
                        percent = min(100, (read_so_far * 100) / total_size)
                        sys.stdout.write(
                            f"\rDownloading: {percent:.1f}% ({read_so_far / (1024 * 1024):.2f} MB / {total_size / (1024 * 1024):.2f} MB)"
                        )
                    else:
                        sys.stdout.write(
                            f"\rDownloading: {read_so_far / (1024 * 1024):.2f} MB (Total size unknown)"
                        )
                    sys.stdout.flush()

        print("\n[+] Download completed successfully!")

    except Exception as e:
        print(f"\n[-] Error occurred during download: {e}")
        print(
            "[!] Please check if the Google Drive link is public ('Anyone with the link')."
        )


if __name__ == "__main__":
    GOOGLE_DRIVE_LINK = input("Google Drive Link: ")
    SAVE_PATH = "./downloaded_file.zip"

    download_from_google_drive(GOOGLE_DRIVE_LINK, SAVE_PATH)
