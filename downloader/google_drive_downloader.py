import os
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

    # 최신 Direct Download URL
    download_url = f"https://drive.usercontent.google.com/download?id={file_id}&confirm=t"

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    try:
        print("[*] Starting download...")
        response = session.get(download_url, stream=True)
        response.raise_for_status()

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

        if os.path.getsize(output_filename) < 1000:  # 다운로드된 파일이 너무 작으면 (HTML 오류 페이지일 가능성)
            print(
                "\n[-] Warning: Downloaded file size is unusually small. Check if permissions are set to 'Anyone with the link'.")
        else:
            print("\n[+] Download completed successfully!")

    except Exception as e:
        print(f"\n[-] Error occurred during download: {e}")


if __name__ == "__main__":
    GOOGLE_DRIVE_LINK = input("Google Drive Link: ")
    SAVE_PATH = "./downloaded_file.zip"

    download_from_google_drive(GOOGLE_DRIVE_LINK, SAVE_PATH)