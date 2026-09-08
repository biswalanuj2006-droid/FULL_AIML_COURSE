import React, { useRef, useState } from 'react'
import { api } from '../lib/api.js'
import { useAuth } from '../auth.jsx'

export default function ChatWidget() {
  const { user } = useAuth()
  const [open, setOpen] = useState(false)
  const [msgs, setMsgs] = useState([{ role: 'assistant', content: "Hi! I'm your AI tutor. Ask me anything, or say 'quiz me on transformers'." }])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const boxRef = useRef(null)

  const send = async (text) => {
    const message = (text ?? input).trim()
    if (!message || busy) return
    if (!user) { setMsgs((m) => [...m, { role: 'assistant', content: 'Sign in first (free) — then I can tutor, quiz and route you to courses.' }]); return }
    setMsgs((m) => [...m, { role: 'user', content: message }])
    setInput(''); setBusy(true)
    try {
      const d = await api('/api/chat', { method: 'POST', body: { message, history: msgs.slice(-8) } })
      setMsgs((m) => [...m, { role: 'assistant', content: d.reply, source: d.source, courses: d.suggested_courses }])
    } catch (e) {
      setMsgs((m) => [...m, { role: 'assistant', content: `Error: ${e.message}` }])
    }
    setBusy(false)
    setTimeout(() => boxRef.current?.scrollTo({ top: 1e9, behavior: 'smooth' }), 50)
  }

  return (
    <div className="fixed bottom-5 right-5 z-50 flex flex-col items-end gap-3">
      {open && (
        <div className="w-[22rem] sm:w-96 h-[30rem] flex flex-col rounded-2xl border border-night-600 bg-night-800 shadow-2xl overflow-hidden">
          <div className="px-4 py-3 bg-night-700 font-semibold flex items-center justify-between">
            <span>AI Tutor <span className="text-xs text-slate-400">answers for all courses</span></span>
            <button onClick={() => setOpen(false)} className="text-slate-400 hover:text-white">✕</button>
          </div>
          <div ref={boxRef} className="flex-1 overflow-y-auto p-3 space-y-3">
            {msgs.map((m, i) => (
              <div key={i} className={m.role === 'user' ? 'text-right' : ''}>
                <div className={`inline-block max-w-[85%] text-sm rounded-2xl px-3 py-2 whitespace-pre-wrap ${m.role === 'user' ? 'bg-accent text-night-900' : 'bg-night-700 text-slate-200'}`}>
                  {m.content}
                </div>
                {m.courses?.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {m.courses.map((c) => (
                      <a key={c} href={`/courses/${c}`} className="badge border border-accent/40 text-accent hover:bg-night-700">open /{c}</a>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {busy && <div className="text-xs text-slate-400 animate-pulse">tutor is thinking…</div>}
          </div>
          <div className="p-3 border-t border-night-600 flex gap-2">
            <input className="input" value={input} placeholder="Ask anything… or 'quiz me on RAG'"
                   onChange={(e) => setInput(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && send()} />
            <button className="btn !px-3" onClick={() => send()} disabled={busy}>➤</button>
          </div>
        </div>
      )}
      <button onClick={() => setOpen((o) => !o)}
              className="w-14 h-14 rounded-full bg-accent text-night-900 text-2xl font-bold shadow-xl hover:scale-105 transition">
        ?
      </button>
    </div>
  )
}
