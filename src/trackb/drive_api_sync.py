from __future__ import annotations

import hashlib
import importlib
import mimetypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DRIVE_FOLDER_MIME = 'application/vnd.google-apps.folder'
DRIVE_DEFAULT_SCOPES = ['https://www.googleapis.com/auth/drive']


def _ensure_google_api_deps() -> None:
    required_modules = [
        'googleapiclient.discovery',
        'googleapiclient.http',
        'google.auth',
    ]
    missing = []
    for mod in required_modules:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            missing.append(mod)

    if not missing:
        return

    cmd = [
        sys.executable,
        '-m',
        'pip',
        'install',
        '--quiet',
        'google-api-python-client',
        'google-auth-httplib2',
        'google-auth-oauthlib',
    ]
    subprocess.run(cmd, check=True)


def _escape_q_literal(text: str) -> str:
    return str(text).replace('\\', '\\\\').replace("'", "\\'")


def _coerce_path_parts(path_or_parts: Sequence[str] | str) -> List[str]:
    if isinstance(path_or_parts, str):
        raw_parts = path_or_parts.replace('\\', '/').split('/')
    else:
        raw_parts = [str(x) for x in path_or_parts]
    return [p.strip() for p in raw_parts if str(p).strip()]


def _list_children(service: Any, parent_id: str) -> List[Dict[str, Any]]:
    q = f"trashed=false and '{_escape_q_literal(parent_id)}' in parents"
    fields = 'nextPageToken, files(id,name,mimeType,md5Checksum,size,modifiedTime)'
    out: List[Dict[str, Any]] = []
    page_token = None
    while True:
        resp = (
            service.files()
            .list(
                q=q,
                fields=fields,
                pageSize=1000,
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        out.extend(resp.get('files', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return out


def _find_folder(service: Any, parent_id: str, name: str) -> Optional[Dict[str, Any]]:
    q = (
        f"trashed=false and '{_escape_q_literal(parent_id)}' in parents "
        f"and mimeType='{DRIVE_FOLDER_MIME}' and name='{_escape_q_literal(name)}'"
    )
    resp = (
        service.files()
        .list(
            q=q,
            fields='files(id,name,mimeType)',
            pageSize=2,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = resp.get('files', [])
    return files[0] if files else None


def _create_folder(service: Any, parent_id: str, name: str) -> str:
    body = {
        'name': str(name),
        'mimeType': DRIVE_FOLDER_MIME,
        'parents': [str(parent_id)],
    }
    resp = (
        service.files()
        .create(
            body=body,
            fields='id',
            supportsAllDrives=True,
        )
        .execute()
    )
    return str(resp['id'])


def ensure_drive_folder_path(
    service: Any,
    path_or_parts: Sequence[str] | str,
    parent_id: str = 'root',
) -> str:
    current = str(parent_id)
    for name in _coerce_path_parts(path_or_parts):
        existing = _find_folder(service, current, name)
        if existing is None:
            current = _create_folder(service, current, name)
        else:
            current = str(existing['id'])
    return current


def _md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download_file(service: Any, file_id: str, dst_path: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open('wb') as out_f:
        req = service.files().get_media(fileId=str(file_id), supportsAllDrives=True)
        downloader = MediaIoBaseDownload(out_f, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def _upload_new_file(service: Any, parent_id: str, local_path: Path) -> None:
    from googleapiclient.http import MediaFileUpload

    mime, _ = mimetypes.guess_type(str(local_path))
    media = MediaFileUpload(
        str(local_path),
        mimetype=(mime or 'application/octet-stream'),
        resumable=True,
    )
    body = {
        'name': local_path.name,
        'parents': [str(parent_id)],
    }
    (
        service.files()
        .create(
            body=body,
            media_body=media,
            fields='id',
            supportsAllDrives=True,
        )
        .execute()
    )


def _update_existing_file(service: Any, file_id: str, local_path: Path) -> None:
    from googleapiclient.http import MediaFileUpload

    mime, _ = mimetypes.guess_type(str(local_path))
    media = MediaFileUpload(
        str(local_path),
        mimetype=(mime or 'application/octet-stream'),
        resumable=True,
    )
    (
        service.files()
        .update(
            fileId=str(file_id),
            media_body=media,
            fields='id',
            supportsAllDrives=True,
        )
        .execute()
    )


def authenticate_drive_api(
    expected_email: Optional[str] = None,
    scopes: Optional[Sequence[str]] = None,
) -> Tuple[Any, str]:
    if not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Drive API auth helper is intended for Colab runtimes.')

    _ensure_google_api_deps()

    try:
        from google.colab import auth as colab_auth
    except Exception as e:
        raise RuntimeError('google.colab.auth is unavailable; run this in Colab.') from e

    colab_auth.authenticate_user()

    import google.auth
    from googleapiclient.discovery import build

    auth_scopes = list(scopes) if scopes else list(DRIVE_DEFAULT_SCOPES)
    creds, _ = google.auth.default(scopes=auth_scopes)
    service = build('drive', 'v3', credentials=creds, cache_discovery=False)
    about = service.about().get(fields='user(emailAddress,displayName)').execute()
    user = about.get('user', {})
    email = str(user.get('emailAddress', '')).strip().lower()
    if not email:
        raise RuntimeError('Unable to resolve authenticated Drive account email.')

    if expected_email and (email != str(expected_email).strip().lower()):
        raise RuntimeError(
            f'Authenticated Drive account mismatch: got {email}, expected {expected_email}.'
        )

    return service, email


def sync_drive_folder_to_local(
    service: Any,
    remote_folder_id: str,
    local_folder: str | Path,
) -> Dict[str, int]:
    local_root = Path(local_folder).expanduser()
    local_root.mkdir(parents=True, exist_ok=True)
    stats = {
        'folders_seen': 0,
        'files_downloaded': 0,
        'files_skipped': 0,
    }

    def _walk(remote_id: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        stats['folders_seen'] += 1
        children = sorted(_list_children(service, remote_id), key=lambda x: (x.get('mimeType', ''), x.get('name', '')))
        for child in children:
            name = str(child.get('name', ''))
            if not name:
                continue
            mime = str(child.get('mimeType', ''))
            if mime == DRIVE_FOLDER_MIME:
                _walk(str(child['id']), local_dir / name)
                continue

            local_path = local_dir / name
            remote_md5 = str(child.get('md5Checksum', '') or '')
            if local_path.exists() and remote_md5:
                try:
                    if _md5sum(local_path) == remote_md5:
                        stats['files_skipped'] += 1
                        continue
                except Exception:
                    pass
            _download_file(service, str(child['id']), local_path)
            stats['files_downloaded'] += 1

    _walk(str(remote_folder_id), local_root)
    return stats


def sync_local_tree_to_drive(
    service: Any,
    local_folder: str | Path,
    remote_folder_id: str,
) -> Dict[str, int]:
    local_root = Path(local_folder).expanduser()
    local_root.mkdir(parents=True, exist_ok=True)
    stats = {
        'folders_seen': 0,
        'files_uploaded': 0,
        'files_updated': 0,
        'files_skipped': 0,
    }
    folder_id_cache: Dict[Path, str] = {Path('.'): str(remote_folder_id)}

    for dirpath, dirnames, filenames in os.walk(local_root):
        dirnames.sort()
        filenames.sort()
        abs_dir = Path(dirpath)
        rel_dir = abs_dir.relative_to(local_root)
        rel_key = Path('.') if str(rel_dir) == '.' else rel_dir
        stats['folders_seen'] += 1

        if rel_key not in folder_id_cache:
            parent_rel = rel_key.parent if str(rel_key.parent) != '' else Path('.')
            parent_id = folder_id_cache[parent_rel]
            folder_id_cache[rel_key] = ensure_drive_folder_path(service, [rel_key.name], parent_id=parent_id)

        current_remote_id = folder_id_cache[rel_key]

        for d in dirnames:
            child_rel = (rel_key / d) if rel_key != Path('.') else Path(d)
            folder_id_cache[child_rel] = ensure_drive_folder_path(service, [d], parent_id=current_remote_id)

        remote_files: Dict[str, Dict[str, Any]] = {}
        for item in _list_children(service, current_remote_id):
            if str(item.get('mimeType', '')) != DRIVE_FOLDER_MIME:
                remote_files[str(item.get('name', ''))] = item

        for fname in filenames:
            local_path = abs_dir / fname
            if not local_path.is_file():
                continue
            local_md5 = _md5sum(local_path)
            remote_item = remote_files.get(fname)
            remote_md5 = str((remote_item or {}).get('md5Checksum', '') or '')
            if remote_item and remote_md5 and remote_md5 == local_md5:
                stats['files_skipped'] += 1
                continue

            if remote_item:
                _update_existing_file(service, str(remote_item['id']), local_path)
                stats['files_updated'] += 1
            else:
                _upload_new_file(service, current_remote_id, local_path)
                stats['files_uploaded'] += 1

    return stats


def initialize_drive_api_sync(
    local_persist_root: str,
    run_tag: str,
    remote_root_folder: str,
    expected_email: Optional[str] = None,
) -> Dict[str, Any]:
    service, email = authenticate_drive_api(expected_email=expected_email)
    remote_root_id = ensure_drive_folder_path(service, remote_root_folder, parent_id='root')
    remote_run_id = ensure_drive_folder_path(service, [run_tag], parent_id=remote_root_id)

    local_root = Path(local_persist_root).expanduser()
    local_run_root = local_root / str(run_tag)
    pull_stats = sync_drive_folder_to_local(
        service=service,
        remote_folder_id=remote_run_id,
        local_folder=local_run_root,
    )

    return {
        'service': service,
        'email': email,
        'run_tag': str(run_tag),
        'remote_root_folder': str(remote_root_folder),
        'remote_root_id': str(remote_root_id),
        'remote_run_id': str(remote_run_id),
        'local_persist_root': str(local_root),
        'local_run_root': str(local_run_root),
        'last_pull_stats': pull_stats,
    }


def pull_drive_api_run(sync_ctx: Dict[str, Any]) -> Dict[str, int]:
    stats = sync_drive_folder_to_local(
        service=sync_ctx['service'],
        remote_folder_id=sync_ctx['remote_run_id'],
        local_folder=sync_ctx['local_run_root'],
    )
    sync_ctx['last_pull_stats'] = stats
    return stats


def push_drive_api_run(sync_ctx: Dict[str, Any]) -> Dict[str, int]:
    stats = sync_local_tree_to_drive(
        service=sync_ctx['service'],
        local_folder=sync_ctx['local_run_root'],
        remote_folder_id=sync_ctx['remote_run_id'],
    )
    sync_ctx['last_push_stats'] = stats
    return stats
