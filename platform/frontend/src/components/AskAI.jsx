import React, { useState } from 'react'
import { api, assetUrl } from '../lib/api.js'
import { useAuth } from '../auth.jsx'
import { Link } from 'react-router-dom'

export default function AskAI({ course = '' }) {
  const { user } = useAuth()
  const [q, setQ] = useState('')
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

  const ask = async () => {
    if (!q.trim() || busy) return
    if (!user) { setErr('Sign in to use the AI answer engine (free).'); return }
    setBusy(true); setErr(''); setRes(null)
    try {
      setRes(await api('/api/answer', { method: 'POST', body: { question: q, course } }))
    } catch (e) { setErr(e.message) }
    setBusy(false)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-bold">🤖 Ask AI about {course ? 'this course' : 'any course'}</h3>
        {res && <span className="badge border border-accent/40 text-accent">confidence {(res.confidence * 100).toFixed(0)}%</span>}
      </div>
      <div className="flex gap-2">
        <input className="input" value={q} placeholder="e.g. why does overfitting happen?"
               onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && ask()} />
        <button className="btn" onClick={ask} disabled={busy}>{busy ? '…' : 'Ask'}</button>
      </div>
      {err && <p className="mt-3 text-sm text-rose-400">{err}</p>}
      {res && (
        <div className="mt-4">
          <p className="text-sm whitespace-pre-wrap leading-relaxed">{res.answer}</p>
          {res.sources?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {res.sources.map((s, i) => (
                <Link key={i} to={`/courses/${s.course}`}
                      className="badge border border-night-600 text-slate-300 hover:border-accent">
                  📚 {s.course_title}: {s.title} ({s.kind})
                </Link>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
