import React, { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import { useAuth } from '../auth.jsx'

export default function Admin() {
  const { user } = useAuth()
  const [stats, setStats] = useState(null)
  const [users, setUsers] = useState([])
  const [subs, setSubs] = useState([])
  const [err, setErr] = useState('')
  const [tab, setTab] = useState('stats')

  const load = async () => {
    try {
      setStats(await api('/api/admin/stats'))
      setUsers(await api('/api/admin/users'))
      setSubs(await api('/api/admin/submissions'))
    } catch (e) { setErr(e.message) }
  }
  useEffect(() => { if (user?.role === 'admin') load() }, [user])

  if (!user) return <p className="text-slate-400">Sign in as an admin.</p>
  if (user.role !== 'admin') return <p className="text-rose-400">Admin access only.</p>
  if (err) return <p className="text-rose-400">{err}</p>

  const setRole = async (uid, role) => {
    await api(`/api/admin/users/${uid}/role`, { method: 'PATCH', body: { role } })
    load()
  }
  const review = async (sid, status) => {
    await api(`/api/admin/submissions/${sid}`, { method: 'PATCH', body: { status, feedback: status === 'passed' ? 'Looks good — nice work!' : 'See notes — revisit edge cases.' } })
    load()
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-extrabold">🛠 Admin panel</h1>
      <div className="flex gap-2">
        {['stats', 'users', 'submissions'].map((t) => (
          <button key={t} onClick={() => setTab(t)} className={t === tab ? 'btn' : 'btn-ghost'}>{t}</button>
        ))}
      </div>

      {tab === 'stats' && stats && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(stats).map(([k, v]) => (
            <div key={k} className="card text-center">
              <div className="text-3xl font-extrabold text-accent">{v}</div>
              <div className="text-sm text-slate-400 mt-1">{k}</div>
            </div>
          ))}
        </div>
      )}

      {tab === 'users' && (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-slate-400 text-left"><tr><th className="py-2">ID</th><th>Email</th><th>Name</th><th>Level</th><th>XP</th><th>Role</th><th></th></tr></thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-t border-night-600">
                  <td className="py-2">{u.id}</td>
                  <td>{u.email}</td><td>{u.name}</td>
                  <td><span className="badge bg-night-700">{u.level}</span></td>
                  <td>{u.xp}</td>
                  <td><span className={`badge ${u.role === 'admin' ? 'bg-fuchsia-500/20 text-fuchsia-300' : 'bg-night-700'}`}>{u.role}</span></td>
                  <td className="text-right">
                    <button className="btn-ghost !py-1 !px-2 text-xs" onClick={() => setRole(u.id, u.role === 'admin' ? 'student' : 'admin')}>
                      {u.role === 'admin' ? 'demote' : 'promote'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {tab === 'submissions' && (
        <div className="space-y-4">
          {subs.length === 0 && <p className="text-slate-500">No pending submissions.</p>}
          {subs.map((s) => (
            <div key={s.id} className="card">
              <div className="flex justify-between items-center">
                <div className="font-semibold">#{s.id} · {s.assignment} <span className="text-slate-500 text-sm">(user {s.user_id})</span></div>
                <div className="flex gap-2">
                  <button className="btn !py-1 text-xs" onClick={() => review(s.id, 'passed')}>approve</button>
                  <button className="btn-ghost !py-1 text-xs" onClick={() => review(s.id, 'feedback')}>needs work</button>
                </div>
              </div>
              <pre className="mt-3 bg-night-900 rounded-xl p-3 text-xs overflow-x-auto text-slate-300">{s.code}</pre>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
