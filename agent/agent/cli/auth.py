#!/usr/bin/env python3
"""
rexgraph-auth - CLI authentication and server management.

Enable auth (turns the API from open to token-gated):
    rexgraph-auth create --name admin --role admin --save   # make a token first
    rexgraph-auth enable                                     # prompts to set a disable passphrase
    rexgraph-auth status                                     # check state (no token)

Turning auth back off is deliberately hard: it must be run on the server host
and requires the disable passphrase (a stolen API token alone is not enough).
    rexgraph-auth passphrase                                 # set/rotate it (host-local)
    rexgraph-auth disable                                    # prompts for the passphrase (host-local)

Token management:
    rexgraph-auth create --name "batch-job" [--role write] [--workspaces a,b]
    rexgraph-auth list
    rexgraph-auth test [--url https://localhost:8000]

Credential storage:
    rexgraph-auth login --url https://hpc.university.edu:8000 --token <token>
    rexgraph-auth whoami
    rexgraph-auth logout

TLS certificates:
    rexgraph-auth gen-cert [--out certs/]
    rexgraph-auth gen-cert --domain hpc.university.edu

Usage with other CLI tools:
    rexgraph-run paper.pdf --server https://localhost:8000
    # Uses stored credentials from 'rexgraph-auth login'
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

def _cred_file() -> Path:
    """Where CLI credentials live, honouring REXGRAPH_CONFIG_DIR.

    Resolved per call rather than at import. It was a module constant over `Path.home()`,
    so it ignored the config directory everything else respects: a test that redirected
    REXGRAPH_CONFIG_DIR still read the operator's real credentials, found the URL of their
    live server, connected to it, and reported that `create` works without a server. It
    also wrote tokens into their home while doing so.
    """
    return Path(os.environ.get(
        "REXGRAPH_CONFIG_DIR", str(Path.home() / ".config" / "rexgraph"))
    ) / "credentials.json"


# Credential storage

def _load_creds() -> dict:
    if _cred_file().exists():
        return json.loads(_cred_file().read_text())
    return {}


def _save_creds(creds: dict):
    _cred_file().parent.mkdir(parents=True, exist_ok=True)
    _cred_file().write_text(json.dumps(creds, indent=2))
    os.chmod(str(_cred_file()), 0o600)  # owner-only


def get_stored_auth() -> tuple:
    """Return (url, token) from stored credentials. Used by other CLI tools."""
    creds = _load_creds()
    return creds.get("url", ""), creds.get("token", "")


# Server communication

def _request(method, url, token=None, json_data=None, verify=True):
    """Make an HTTP request to the rexgraph server."""
    import ssl
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = None
    if json_data is not None:
        data = json.dumps(json_data).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    ctx = None
    if not verify or url.startswith("https://localhost") or url.startswith("https://127."):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}
    except Exception as e:
        return 0, {"error": str(e)}


def _report_failure(status, data, url=None):
    """Print why a request failed, and what to do about it.

    `_request` returns status 0 when it never reached the server at all, which is a
    different problem from a 401 and needs a different answer. Every command printed
    `Failed (0): <urlopen error [Errno 111] Connection refused>` for it, which names
    the errno and not one thing the caller can act on.
    """
    if status == 0:
        where = url or "the server"
        print(f"Cannot reach {where}: {data.get('error', 'connection failed')}",
              file=sys.stderr)
        print("This command talks to a running server. Start one with "
              "`rexgraph-serve`, point at another with --url, or store credentials "
              "with `rexgraph-auth login --url URL --admin-token TOKEN`.",
              file=sys.stderr)
        return
    print("Failed (%d): %s" % (status, data.get("error", data)), file=sys.stderr)


# Commands

def cmd_create(args):
    """Create a new API token."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = args.admin_token or _load_creds().get("token", "")

    workspaces = [w.strip() for w in (args.workspaces or "default").split(",") if w.strip()]
    # NOTE: endpoint is /token (singular); the server body is user_id/workspaces/role.
    status, data = _request("POST", f"{url}/api/v1/admin/token",
                            token=token,
                            json_data={"user_id": args.name,
                                       "workspaces": workspaces,
                                       "role": args.role})

    if status == 200 and "token" in data:
        print("Token created: {}".format(data["token"]))
        print("User: {}  role: {}  workspaces: {}".format(
            args.name, args.role, ",".join(workspaces)))
        if args.save:
            creds = _load_creds()
            creds["url"] = url
            creds["token"] = data["token"]
            _save_creds(creds)
            print(f"Saved to {_cred_file()}")
    else:
        _report_failure(status, data, url)
        if status in (401, 403):
            print("Hint: while auth is disabled any local caller is admin; once "
                  "enabled, pass an admin token via --admin-token or "
                  "'rexgraph-auth login'.", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    """List all API tokens."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = _load_creds().get("token", "")

    status, data = _request("GET", f"{url}/api/v1/admin/tokens", token=token)

    if status == 200:
        tokens = data.get("tokens", [])
        if not tokens:
            print("No tokens configured (auth disabled)")
            return
        for t in tokens:
            name = t.get("name", "unnamed")
            created = t.get("created", "?")
            prefix = t.get("prefix", "?")
            print(f"  {prefix}...  name={name}  created={created}")
    else:
        _report_failure(status, data, url)


def cmd_test(args):
    """Test connection and authentication against a running server."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = _load_creds().get("token", "")

    print(f"Testing: {url}")

    # Health check (no auth)
    status, data = _request("GET", f"{url}/api/health")
    if status == 200:
        print("  Health: OK ({})".format(data.get("status", "?")))
    else:
        print("  Health: FAILED (%d)" % status)
        sys.exit(1)

    # Auth check
    status, data = _request("GET", f"{url}/api/v1/admin/tokens", token=token)
    if status == 200:
        n = len(data.get("tokens", []))
        enabled = data.get("auth_enabled", n > 0)
        print("  Auth: %s (%d tokens)" % (
            "enabled" if enabled else "disabled", n))
    elif status == 401:
        print("  Auth: enabled (invalid or missing token)")
    else:
        print("  Auth: unknown (%d)" % status)

    # TLS check
    if url.startswith("https://"):
        print("  TLS: yes")
    else:
        print("  TLS: no (use --url https://... for secure connections)")

    # Model status
    status, data = _request("GET", f"{url}/api/v1/models/status",
                            token=token)
    if status == 200:
        print("  Models: %d loaded, %d available" % (
            data.get("n_loaded", 0), data.get("n_available", 0)))


def _prompt_passphrase(confirm: bool) -> str:
    """Read a passphrase from the terminal without echoing it."""
    p = getpass.getpass("Disable passphrase: ")
    if confirm:
        again = getpass.getpass("Confirm passphrase: ")
        if p != again:
            print("Passphrases do not match.", file=sys.stderr)
            sys.exit(1)
    return p


def cmd_enable(args):
    """Enable bearer-token authentication on the server (run on the host)."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = args.admin_token or _load_creds().get("token", "")
    # Set the disable passphrase at enable time so auth can later be turned off.
    passphrase = _prompt_passphrase(confirm=True)
    status, data = _request("POST", f"{url}/api/v1/admin/auth/enable", token=token,
                            json_data={"passphrase": passphrase})
    if status == 200 and data.get("auth_enabled"):
        print("Authentication ENABLED - all API calls now require a bearer token.")
        print("Disable passphrase set: {}".format(data.get("disable_passphrase_set", False)))
        if not token:
            print("You have no stored admin token. Create one BEFORE you lose access:")
            print("  rexgraph-auth create --name admin --role admin --save")
    else:
        _report_failure(status, data, url)
        if status in (401, 403):
            print("Hint: run this on the server host, and pass an admin token via "
                  "--admin-token or 'rexgraph-auth login'.", file=sys.stderr)
        sys.exit(1)


def cmd_passphrase(args):
    """Set or rotate the passphrase required to disable auth (run on the host)."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = args.admin_token or _load_creds().get("token", "")
    passphrase = _prompt_passphrase(confirm=True)
    status, data = _request("POST", f"{url}/api/v1/admin/auth/passphrase", token=token,
                            json_data={"passphrase": passphrase})
    if status == 200 and data.get("disable_passphrase_set"):
        print("Disable passphrase set.")
    else:
        _report_failure(status, data, url)
        sys.exit(1)


def cmd_disable(args):
    """Disable authentication (open access). Host-local and passphrase-gated."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = args.admin_token or _load_creds().get("token", "")
    passphrase = _prompt_passphrase(confirm=False)
    status, data = _request("POST", f"{url}/api/v1/admin/auth/disable", token=token,
                            json_data={"passphrase": passphrase})
    if status == 200 and data.get("auth_enabled") is False:
        print("Authentication DISABLED - the API is now open. Do not expose the "
              "port (bind 127.0.0.1, or set RCF_ALLOW_INSECURE=1 knowingly).")
    else:
        _report_failure(status, data, url)
        if status == 403:
            print("Hint: disabling auth requires running on the server host and the "
                  "disable passphrase. Set one with 'rexgraph-auth passphrase'.",
                  file=sys.stderr)
        sys.exit(1)


def cmd_status(args):
    """Show whether auth is enabled on a server (no token required)."""
    creds = _load_creds()
    url = args.url or creds.get("url", "http://localhost:8000")
    status, data = _request("GET", f"{url}/api/health")
    if status != 200:
        # a status query answers with what it knows. "No server there" IS the answer,
        # so it prints the local half and exits 0; exiting 1 made `status` unusable as
        # the first command someone runs, before anything is serving.
        print(f"Server:     {url}")
        print("Reachable:  no (%s)" % (data.get("error", status)
                                       if status == 0 else f"HTTP {status}"))
        print("Credentials: %s" % ("stored for " + creds["url"]
                                   if creds.get("url") else "none stored"))
        print("Start one with `rexgraph-serve`, or point at another with --url.")
        return
    print(f"Server:     {url}")
    print("Auth:       %s" % ("ENABLED" if data.get("auth_enabled") else "disabled (open)"))
    print("Backend:    rexgraph {}".format(data.get("rexgraph", "?")))
    print("Workspaces: {}".format(", ".join(data.get("workspaces", []) or ["default"])))


def cmd_login(args):
    """Store credentials for CLI access to a server."""
    creds = {
        "url": args.url,
        "token": args.token,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_creds(creds)
    print(f"Credentials saved to {_cred_file()}")
    print(f"  URL: {args.url}")
    print(f"  Token: {args.token[:8]}...{args.token[-4:]}"
          if len(args.token) > 12 else "  Token: (short)")

    # Test the connection
    status, data = _request("GET", f"{args.url}/api/health")
    if status == 200:
        print("  Connection: OK")
    else:
        print(f"  Warning: could not connect to {args.url}")


def cmd_whoami(args):
    """Show stored credentials."""
    creds = _load_creds()
    if not creds:
        print("No credentials stored. Run: rexgraph-auth login")
        return
    print("  URL: {}".format(creds.get("url", "(not set)")))
    token = creds.get("token", "")
    if token:
        print(f"  Token: {token[:8]}...{token[-4:]}"
              if len(token) > 12 else f"  Token: {token}")
    print("  Saved: {}".format(creds.get("saved_at", "?")))


def cmd_logout(args):
    """Remove stored credentials."""
    if _cred_file().exists():
        _cred_file().unlink()
        print("Credentials removed.")
    else:
        print("No credentials stored.")


def cmd_gen_cert(args):
    """Generate a self-signed TLS certificate for development/HPC."""
    import subprocess

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    key_path = out_dir / "server.key"
    cert_path = out_dir / "server.crt"

    domain = args.domain or "localhost"
    days = args.days or 365

    # Generate with OpenSSL
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", str(key_path),
        "-out", str(cert_path),
        "-days", str(days),
        "-nodes",  # no passphrase
        "-subj", f"/CN={domain}",
        "-addext", f"subjectAltName=DNS:{domain},DNS:localhost,IP:127.0.0.1",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        print("Error: openssl not found. Install with: sudo apt install openssl",
              file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

    os.chmod(str(key_path), 0o600)

    print("TLS certificate generated:")
    print(f"  Key:  {key_path}")
    print(f"  Cert: {cert_path}")
    print("")
    print("Start server with TLS:")
    print(f"  make serve SSL_KEY={key_path} SSL_CERT={cert_path}")
    print("")
    print("Or directly:")
    print(f"  cd agent && python run.py --ssl-cert {cert_path} --ssl-key {key_path}")
    print("")
    print("Or via the console script (rcf-server):")
    print(f"  REXGRAPH_TLS_CERT={cert_path} REXGRAPH_TLS_KEY={key_path} rcf-server")


def cmd_member(args):
    """Manage a workspace's roster: add/update a member, list members, or revoke access.

    Roles are per workspace ('--workspace', default 'default'): 'user' (read + build verbs) or 'admin'
    (also manages that workspace's members and runs its consequential actions). 'add' mints a token
    for a new member (shown once) or updates an existing member's role in place (their token is kept).
    You must be an admin OF THAT WORKSPACE (instance admin = admin of 'default')."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = args.admin_token or _load_creds().get("token", "")
    ws = args.workspace or "default"
    if args.action == "add":
        if not args.name:
            print("--name is required for 'member add'", file=sys.stderr)
            sys.exit(1)
        status, data = _request("POST", f"{url}/api/v1/admin/members?workspace={ws}", token=token,
                                json_data={"user_id": args.name, "role": args.role})
        if status == 200 and data.get("token"):
            print("Member added to '{}': {}  role: {}".format(ws, args.name, data.get("role", args.role)))
            print("Token (give it to them, shown once): {}".format(data["token"]))
        elif status == 200:
            print("Member '{}' updated in '{}': role {} (their existing token is unchanged).".format(
                args.name, ws, data.get("role", args.role)))
        else:
            _report_failure(status, data, url)
            sys.exit(1)
    elif args.action == "list":
        status, data = _request("GET", f"{url}/api/v1/admin/members?workspace={ws}", token=token)
        if status == 200:
            members = data.get("members", [])
            print("workspace: {}".format(data.get("workspace", ws)))
            if not members:
                print("  (no members)")
                return
            print("  %-20s %-8s" % ("USER", "ROLE"))
            for m in members:
                print("  %-20s %-8s" % (m.get("user_id", "?"), m.get("role", "?")))
        else:
            _report_failure(status, data, url)
            sys.exit(1)
    elif args.action == "revoke":
        if not args.name:
            print("--name is required for 'member revoke'", file=sys.stderr)
            sys.exit(1)
        q = "?workspace={}{}".format(ws, "&all=true" if args.all else "")
        status, data = _request("DELETE", f"{url}/api/v1/admin/members/{args.name}{q}",
                                token=token)
        if status == 200:
            where = "everywhere" if args.all else ("workspace '{}'".format(data.get("workspace", ws)))
            print(f"Revoked: {args.name} from {where}")
        else:
            _report_failure(status, data, url)
            sys.exit(1)


def cmd_network_init(args):
    """Stand up the shared multi-user posture in one step: the default for a networked deployment.

    Creates the first admin, a recovery key, and turns auth on with a disable passphrase. Run it on
    the server host while auth is still off (a local caller is admin until then). Solo/local use needs
    none of this - leaving auth off keeps the single local-admin identity."""
    url = args.url or _load_creds().get("url", "http://localhost:8000")
    token = args.admin_token or _load_creds().get("token", "")
    name = args.name or "admin"

    status, data = _request("POST", f"{url}/api/v1/admin/members?workspace=default", token=token,
                            json_data={"user_id": name, "role": "admin"})
    if status != 200 or "token" not in data:
        print("Failed to create the first admin (%d): %s" % (status, data.get("error", data)),
              file=sys.stderr)
        if status in (401, 403):
            print("Hint: run this on the host while auth is still disabled.", file=sys.stderr)
        sys.exit(1)
    admin_token = data["token"]

    status, rdata = _request("POST", f"{url}/api/v1/admin/recovery-key", token=admin_token)
    recovery = rdata.get("recovery_key") if status == 200 else None

    passphrase = _prompt_passphrase(confirm=True)
    status, edata = _request("POST", f"{url}/api/v1/admin/auth/enable", token=admin_token,
                             json_data={"passphrase": passphrase})
    if status != 200 or not edata.get("auth_enabled"):
        print("Admin created but enabling auth failed (%d): %s" % (status, edata.get("error", edata)),
              file=sys.stderr)
        print(f"Save this admin token now: {admin_token}", file=sys.stderr)
        sys.exit(1)

    creds = _load_creds()
    creds["url"] = url
    creds["token"] = admin_token
    _save_creds(creds)

    print("Shared multi-user network is up. Authentication is ENABLED.")
    print("")
    print(f"  Admin user:    {name}")
    print(f"  Admin token:   {admin_token}")
    print(f"                 (saved to {_cred_file()}; shown once)")
    if recovery:
        print(f"  Recovery key:  {recovery}")
        print("                 (store offline - the only way back if all tokens are lost)")
    print("")
    print("Add members (each gets their own token, shown once):")
    print("  rexgraph-auth member add --name alice --role user")
    print("  rexgraph-auth member add --name bob   --role admin")
    print("Or use the System -> auth panel in the web UI.")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="rexgraph authentication and server management")
    sub = parser.add_subparsers(dest="command")

    # create (low-level; 'member add' is the managed, rotate-per-user path)
    p = sub.add_parser("create", help="Create an API token (low-level; prefer 'member add')")
    p.add_argument("--name", required=True, help="Token user_id/label")
    p.add_argument("--role", default="user", choices=["user", "admin", "read", "write"],
                   help="Token role (default: user; read/write fold to user)")
    p.add_argument("--workspaces", help="Comma-separated workspaces (default: default)")
    p.add_argument("--url", help="Server URL (default: stored or localhost:8000)")
    p.add_argument("--admin-token", help="Admin token for auth")
    p.add_argument("--save", action="store_true", help="Save as default credential")

    # member (managed roster: per-workspace admin/user roles, one token per user)
    p = sub.add_parser("member", help="Manage a workspace's members (add/list/revoke)")
    p.add_argument("action", choices=["add", "list", "revoke"], help="What to do")
    p.add_argument("--name", help="Member user id (for add/revoke)")
    p.add_argument("--role", default="user", choices=["user", "admin"],
                   help="Role for 'add' (default: user)")
    p.add_argument("--workspace", default="default",
                   help="Workspace to manage (default: default = the instance)")
    p.add_argument("--all", action="store_true",
                   help="For 'revoke': remove the member entirely (needs instance admin)")
    p.add_argument("--url", help="Server URL")
    p.add_argument("--admin-token", help="Admin token for auth")

    # network-init (the shared multi-user default for a networked deployment)
    p = sub.add_parser("network-init",
                       help="Stand up shared multi-user auth: first admin + recovery key + auth on")
    p.add_argument("--name", default="admin", help="First admin user id (default: admin)")
    p.add_argument("--url", help="Server URL")
    p.add_argument("--admin-token", help="Admin token (usually unset; auth is still off)")

    # enable
    p = sub.add_parser("enable", help="Enable authentication on the server")
    p.add_argument("--url", help="Server URL")
    p.add_argument("--admin-token", help="Admin token for auth")

    # disable
    p = sub.add_parser("disable", help="Disable authentication (host-local + passphrase)")
    p.add_argument("--url", help="Server URL")
    p.add_argument("--admin-token", help="Admin token for auth")

    # passphrase
    p = sub.add_parser("passphrase", help="Set/rotate the passphrase required to disable auth")
    p.add_argument("--url", help="Server URL")
    p.add_argument("--admin-token", help="Admin token for auth")

    # status
    p = sub.add_parser("status", help="Show whether auth is enabled (no token needed)")
    p.add_argument("--url", help="Server URL")

    # list
    p = sub.add_parser("list", help="List API tokens")
    p.add_argument("--url", help="Server URL")

    # test
    p = sub.add_parser("test", help="Test server connection and auth")
    p.add_argument("--url", help="Server URL")

    # login
    p = sub.add_parser("login", help="Store credentials for CLI access")
    p.add_argument("--url", required=True, help="Server URL (e.g. https://hpc:8000)")
    p.add_argument("--token", required=True, help="API token")

    # whoami
    sub.add_parser("whoami", help="Show stored credentials")

    # logout
    sub.add_parser("logout", help="Remove stored credentials")

    # gen-cert
    p = sub.add_parser("gen-cert", help="Generate self-signed TLS certificate")
    p.add_argument("--out", default="certs", help="Output directory (default: certs/)")
    p.add_argument("--domain", help="Domain name (default: localhost)")
    p.add_argument("--days", type=int, default=365, help="Validity in days")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmd_map = {
        "create": cmd_create,
        "member": cmd_member,
        "network-init": cmd_network_init,
        "enable": cmd_enable,
        "disable": cmd_disable,
        "passphrase": cmd_passphrase,
        "status": cmd_status,
        "list": cmd_list,
        "test": cmd_test,
        "login": cmd_login,
        "whoami": cmd_whoami,
        "logout": cmd_logout,
        "gen-cert": cmd_gen_cert,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
