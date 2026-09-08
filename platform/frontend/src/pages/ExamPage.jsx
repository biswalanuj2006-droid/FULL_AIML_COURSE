import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { api, assetUrl } from '../lib/api.js'
import { useAuth } from '../auth.jsx'
import AskAI from '../components/AskAI.jsx'

export default function ExamPage() {
  const { id } = useParams()
  const { user, refresh } = useAuth()
  const [exam, setExam] = useState(null)
  const [answers, setAnswers] = useState({})
  const [result, setResult] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => { api(`/api/exams/${id}`).then(setExam).catch((e) => setErr(e.message)) }, [id])

  if (err) return <p className="text-rose-400">{err}</p>
  if (!exam) return <p className="text-slate-400 animate-pulse">Loading exam…</p>

  const submit = async () => {
    if (!user) { setErr('Sign in to submit the exam (free).'); return }
    try {
      const r = await api(`/api/exams/${exam.id}/submit`, { method: 'POST', body: { answers } })
      setResult(r); refresh()
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) { setErr(e.message) }
  }

  const answeredCount = Object.keys(answers).length

  return (
    <div className="space-y-6">
      <header>
        <div className="text-sm text-slate-500">Exam</div>
        <h1 className="text-3xl font-extrabold">{exam.title}</h1>
        <p className="text-slate-400 text-sm mt-1">Pass mark {exam.pass_pct}% · {exam.questions.length} questions · visual diagram questions included</p>
      </header>

      {result ? (
        <div className="space-y-6">
          <div className={`card text-center ${result.passed ? 'border-emerald-500/50' : 'border-rose-500/50'}`}>
            <div className="text-5xl font-extrabold">{result.score_pct}%</div>
            <div className={`mt-2 font-bold ${result.passed ? 'text-emerald-400' : 'text-rose-400'}`}>
              {result.passed ? `PASSED — +${result.xp_earned} XP 🎉` : `Not passed — +${result.xp_earned} XP (retry anytime)`}
            </div>
          </div>
          <div className="space-y-4">
            {result.review.map((r, i) => (
              <div key={r.id} className={`card ${r.correct ? 'border-emerald-600/40' : 'border-rose-600/40'}`}>
                <div className="font-semibold">Q{i + 1}. {r.prompt}</div>
                <div className={`text-sm mt-1 ${r.correct ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {r.correct ? '✓ correct' : `✗ correct answer: option ${r.answer_index + 1}`}
                </div>
                {r.explanation && <p className="text-sm text-slate-400 mt-2">{r.explanation}</p>}
              </div>
            ))}
          </div>
          <AskAI course="" />
          <button className="btn-ghost" onClick={() => { setResult(null); setAnswers({}) }}>Retake exam</button>
        </div>
      ) : (
        <>
          <div className="space-y-4">
            {exam.questions.map((q, i) => (
              <div key={q.id} className="card">
                <div className="flex items-start gap-3">
                  <span className="badge bg-night-700 text-slate-300 mt-0.5">{q.kind === 'visual' ? '📊 visual' : 'Q' + (i + 1)}</span>
                  <div className="flex-1">
                    <div className="font-semibold">{q.prompt}</div>
                    {q.diagram && (
                      <img className="mt-3 rounded-xl border border-night-600 max-h-72 w-full object-contain bg-white/5"
                           src={assetUrl(q.diagram)} alt="question diagram"
                           onError={(e) => { e.currentTarget.style.display = 'none' }} />
                    )}
                    <div className="mt-3 space-y-2">
                      {q.options.map((opt, oi) => (
                        <label key={oi} className={`flex items-center gap-3 rounded-xl border px-4 py-2.5 cursor-pointer text-sm
                          ${answers[q.id] === oi ? 'border-accent bg-accent/10' : 'border-night-600 hover:border-slate-500'}`}>
                          <input type="radio" className="accent-teal-400" name={`q${q.id}`}
                                 checked={answers[q.id] === oi} onChange={() => setAnswers({ ...answers, [q.id]: oi })} />
                          {opt}
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="sticky bottom-4 card flex items-center justify-between">
            <span className="text-sm text-slate-400">{answeredCount}/{exam.questions.length} answered</span>
            <button className="btn" onClick={submit} disabled={answeredCount === 0}>Submit exam</button>
          </div>
        </>
      )}
    </div>
  )
}
