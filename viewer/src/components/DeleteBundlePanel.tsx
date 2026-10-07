import { useState } from "react";

interface Entry {
  pkg: string;
  ver: string;
  ingested: boolean;
  archived: boolean;
}

interface Props {
  entries: Entry[];
}

export default function DeleteBundlePanel({ entries }: Props) {
  const [rows, setRows] = useState(entries);
  const [busy, setBusy] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [alsoRaw, setAlsoRaw] = useState(true);

  const onDelete = async (e: Entry) => {
    const what = alsoRaw && e.archived ? "and its raw archive entry " : "";
    if (!window.confirm(`Delete ${e.pkg} ${e.ver} ${what}from the store?`)) return;
    const key = `${e.pkg}@${e.ver}`;
    setBusy(key);
    setResult(null);
    try {
      const params = new URLSearchParams({ pkg: e.pkg, ver: e.ver });
      if (alsoRaw) params.set("raw", "1");
      const resp = await fetch(`/api/delete-bundle?${params}`, { method: "POST" });
      const body = (await resp.json()) as { ok: boolean; error?: string; elapsed_s?: string };
      if (!resp.ok || !body.ok) {
        setResult(`Error: ${body.error ?? `HTTP ${resp.status}`}`);
      } else {
        setResult(`Deleted ${key} in ${body.elapsed_s ?? "?"}s.`);
        setRows((rs) =>
          alsoRaw
            ? rs.filter((r) => r !== e)
            : rs.map((r) => (r === e ? { ...r, ingested: false } : r)).filter((r) => r.archived)
        );
      }
    } catch (err) {
      setResult(`Network error: ${err}`);
    }
    setBusy(null);
  };

  return (
    <div className="clear-graphstore">
      <label>
        <input type="checkbox" checked={alsoRaw} onChange={(ev) => setAlsoRaw(ev.target.checked)} />{" "}
        Also delete the raw archive entry (otherwise a reingest restores the bundle)
      </label>
      {rows.length === 0 ? (
        <p className="admin-empty">No bundles.</p>
      ) : (
        <table className="admin-stats-table">
          <thead>
            <tr>
              <th>Bundle</th>
              <th>Ingested</th>
              <th>Archived</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr key={`${e.pkg}@${e.ver}`}>
                <td>
                  <code>
                    {e.pkg} {e.ver}
                  </code>
                </td>
                <td>{e.ingested ? "yes" : "no"}</td>
                <td>{e.archived ? "yes" : "no"}</td>
                <td>
                  <button
                    className="clear-graphstore-btn"
                    type="button"
                    disabled={busy !== null}
                    onClick={() => onDelete(e)}
                  >
                    {busy === `${e.pkg}@${e.ver}` ? "Deleting…" : "Delete"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {result && (
        <div
          className={`clear-graphstore-result ${result.startsWith("Error") || result.startsWith("Network") ? "clear-graphstore-result--error" : "clear-graphstore-result--ok"}`}
        >
          {result}
        </div>
      )}
    </div>
  );
}
