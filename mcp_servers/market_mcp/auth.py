from __future__ import annotations

import os
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

load_dotenv()

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI")

if not CLIENT_ID:
    raise ValueError("Missing FYERS_CLIENT_ID")

if not SECRET_KEY:
    raise ValueError("Missing FYERS_SECRET_KEY")

if not REDIRECT_URI:
    raise ValueError("Missing FYERS_REDIRECT_URI")


class FyersAuthenticator:

    def __init__(self):

        self.session = fyersModel.SessionModel(
            client_id=CLIENT_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            response_type="code",
            grant_type="authorization_code",
            state="spider_sence",
        )

    def login(self):

        login_url = self.session.generate_authcode()

        print("\nOpening browser...\n")

        print(login_url)

        webbrowser.open(login_url)

        print(
            "\nAfter login copy the COMPLETE redirected URL and paste it below.\n"
        )

        redirected = input("Redirect URL: ").strip()

        parsed = urlparse(redirected)

        params = parse_qs(parsed.query)

        auth_code = params.get("auth_code")

        if not auth_code:
            raise RuntimeError("auth_code not found")

        return auth_code[0]

    def generate_access_token(self, auth_code):

        self.session.set_token(auth_code)

        response = self.session.generate_token()

        if "access_token" not in response:

            raise RuntimeError(response)

        return response["access_token"]


def save_token(token: str):

    env_file = Path(".env")

    lines = env_file.read_text().splitlines()

    found = False

    new_lines = []

    for line in lines:

        if line.startswith("FYERS_ACCESS_TOKEN="):

            new_lines.append(f"FYERS_ACCESS_TOKEN={token}")

            found = True

        else:

            new_lines.append(line)

    if not found:

        new_lines.append(f"FYERS_ACCESS_TOKEN={token}")

    env_file.write_text("\n".join(new_lines) + "\n")


def main():

    auth = FyersAuthenticator()

    auth_code = auth.login()

    print("\nGenerating access token...\n")

    token = auth.generate_access_token(auth_code)

    save_token(token)

    print("\nAccess token saved successfully.\n")


if __name__ == "__main__":
    main()