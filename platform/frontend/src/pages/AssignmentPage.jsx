import React, { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../lib/api.js'
import { useAuth } from '../auth.jsx'
import AskAI from '../components/AskAI.jsx'

export default function AssignmentPage() {
  const { id } = useParams()
  const { user } = useAuth()
  const [a, setA] = useState(null)
  const [code, setCode] = useState('')
  const [msg, setMsg] = useState('')
  const [err, setErr] = useState('')

  useEffect(() => {
    api(`/api/assignments/${id}`).then((d) => { setA(d); setCode(d.starter_code || '') }).catch((e) => setErr(e.message))
  }, [id])

  if (err) return <p className="text-rose-400">{err}</p>
  if (!a) return <p className="text-slate-400 animate-pulse">Loading…</p>

  const submit = async () => {
    if (!user) { setErr('Sign in to submit (free).'); return }
    try {
      const r = await api(`/api/assignments/${a.id}/submit`, { method: 'POST', body: { code } })
      setMsg(r.message)
    } catch (e) { setErr(e.message) }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <header>
        <div className="text-sm text-slate-500">Assignment</div>
        <h1 className="text-3xl font-extrabold">{a.title}</h1>
      </header>
      <div className="card">
        <p className="text-slate-300 whitespace-pre-wrap">{a.brief}</p>
      </div>
      <div className="card">
        <h2 className="font-bold mb-3">Your solution</h2>
        <textarea className="input font-mono text-sm min-h-72" value={code} onChange={(e) => setCode(e.target.value)} />
        {err && <p className="mt-3 text-sm text-rose-400">{err}</p>}
        {msg && <p className="mt-3 text-sm text-emerald-400">{msg}</p>}
        <button className="btn mt-4" onClick={submit}>Submit for review</button>
      </div>
      <AskAI course="" />
    </div>
  )
}
