import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../auth.jsx'

export default function Login() {
  const { login, register } = useAuth()
  const nav = useNavigate()
  const [mode, setMode] = useState('login')
  const [form, setForm] = useState({ email: '', name: '', password: '' })
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setErr('')
    try {
      if (mode === 'login') await login(form.email, form.password)
      else await register(form.email, form.name, form.password)
      nav('/dashboard')
    } catch (ex) { setErr(ex.message) }
    setBusy(false)
  }

  return (
    <div className="max-w-md mx-auto">
      <div className="card">
        <h1 className="text-2xl font-bold mb-1">{mode === 'login' ? 'Welcome back' : 'Create your free account'}</h1>
        <p className="text-sm text-slate-400 mb-6">Anyone can sign up — courses, exams and projects included.</p>
        <form onSubmit={submit} className="space-y-4">
          {mode === 'register' && (
            <input className="input" placeholder="Your name" value={form.name}
                   onChange={(e) => setForm({ ...form, name: e.target.value })} required />
          )}
          <input className="input" type="email" placeholder="Email" value={form.email}
                 onChange={(e) => setForm({ ...form, email: e.target.value })} required />
          <input className="input" type="password" placeholder="Password (min 8 chars)" value={form.password}
                 onChange={(e) => setForm({ ...form, password: e.target.value })} required minLength={8} />
          {err && <p className="text-sm text-rose-400">{err}</p>}
          <button className="btn w-full" disabled={busy}>{busy ? '…' : mode === 'login' ? 'Sign in' : 'Create account'}</button>
        </form>
        <p className="mt-4 text-sm text-slate-400 text-center">
          {mode === 'login' ? (
            <>New here? <button className="text-accent hover:underline" onClick={() => setMode('register')}>Create an account</button></>
          ) : (
            <>Already have one? <button className="text-accent hover:underline" onClick={() => setMode('login')}>Sign in</button></>
          )}
        </p>
        <div className="mt-4 text-xs text-slate-500 bg-night-900 rounded-xl p-3 border border-night-600">
          demo accounts — admin: <code>admin@aiml.dev / admin12345</code> · learner: <code>demo@aiml.dev / demo12345</code>
        </div>
      </div>
    </div>
  )
}
