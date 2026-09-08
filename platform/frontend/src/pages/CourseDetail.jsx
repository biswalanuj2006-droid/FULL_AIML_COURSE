import React, { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, assetUrl } from '../lib/api.js'
import AskAI from '../components/AskAI.jsx'

const DIFF_COLOR = { E: 'bg-emerald-500/20 text-emerald-300', D: 'bg-teal-500/20 text-teal-300', C: 'bg-sky-500/20 text-sky-300', B: 'bg-indigo-500/20 text-indigo-300', A: 'bg-violet-500/20 text-violet-300', S: 'bg-fuchsia-500/20 text-fuchsia-300', 'S+': 'bg-pink-500/20 text-pink-300', 'S++': 'bg-rose-500/20 text-rose-300' }

export default function CourseDetail() {
  const { slug } = useParams()
  const [c, setC] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => { api(`/api/courses/${slug}`).then(setC).catch((e) => setErr(e.message)) }, [slug])

  if (err) return <p className="text-rose-400">{err}</p>
  if (!c) return <p className="text-slate-400 animate-pulse">Loading course…</p>

  return (
    <div className="space-y-10">
      <header>
        <div className="text-sm text-slate-500"><Link to="/courses" className="hover:text-accent">Courses</Link> / {c.slug}</div>
        <h1 className="text-3xl font-extrabold mt-1">{c.title}</h1>
        <p className="text-slate-400 mt-2 max-w-3xl">{c.description}</p>
      </header>

      <AskAI course={c.slug} />

      <section>
        <h2 className="text-2xl font-bold mb-4">Lessons</h2>
        <div className="space-y-4">
          {c.lesson_list.map((l) => (
            <div key={l.id} className="card">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="font-bold">{l.order}. {l.title}</div>
                  <p className="text-sm text-slate-400 mt-1">{l.summary}</p>
                </div>
                <div className="flex gap-2 shrink-0">
                  {l.video && <a className="btn-ghost !py-1 text-xs" href={assetUrl(l.video)} target="_blank" rel="noreferrer">▶ video</a>}
                  {l.diagram && <a className="btn-ghost !py-1 text-xs" href={assetUrl(l.diagram)} target="_blank" rel="noreferrer">🖼 diagram</a>}
                </div>
              </div>
              {l.diagram && (
                <img className="mt-4 rounded-xl border border-night-600 max-h-80 object-contain w-full bg-white/5"
                     src={assetUrl(l.diagram)} alt={l.title}
                     onError={(e) => { e.currentTarget.style.display = 'none' }} />
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-xl font-bold">🎓 Final exam</h2>
          <p className="text-sm text-slate-400 mt-2">Pass at 70% to earn XP. Every question has an explanation — and the AI explains anything you miss.</p>
          {c.exam_ids.map((id) => (
            <Link key={id} to={`/exams/${id}`} className="btn mt-4">Take the exam →</Link>
          ))}
        </div>
        <div className="card">
          <h2 className="text-xl font-bold">📝 Assignments</h2>
          <p className="text-sm text-slate-400 mt-2">Coding tasks with starter code — submit for review.</p>
          <div className="mt-3 space-y-2">
            {c.assignment_ids.map((id) => (
              <Link key={id} to={`/assignments/${id}`} className="block text-accent hover:underline text-sm">Assignment #{id} →</Link>
            ))}
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-2xl font-bold mb-4">🚀 Projects <span className="text-sm font-normal text-slate-500">(difficulty E → S++)</span></h2>
        <div className="grid sm:grid-cols-2 gap-4">
          {c.project_list.map((p) => (
            <div key={p.id} className="card">
              <div className="flex items-center justify-between">
                <div className="font-bold">{p.title}</div>
                <span className={`badge ${DIFF_COLOR[p.difficulty] || 'bg-night-700'}`}>{p.difficulty}</span>
              </div>
              <p className="text-sm text-slate-400 mt-2">{p.objective}</p>
              <div className="mt-3 flex flex-wrap gap-1">
                {(p.tech || []).map((t) => <span key={t} className="badge bg-night-700 text-slate-400">{t}</span>)}
              </div>
            </div>
          ))}
          {c.project_list.length === 0 && <p className="text-slate-500 text-sm">Projects for this course are being designed.</p>}
        </div>
      </section>
    </div>
  )
}
