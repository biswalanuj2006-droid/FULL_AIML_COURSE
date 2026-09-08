import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../lib/api.js'

export default function Courses() {
  const [courses, setCourses] = useState([])
  const [q, setQ] = useState('')
  useEffect(() => { api('/api/courses').then(setCourses).catch(() => {}) }, [])
  const filtered = courses.filter((c) =>
    (c.title + ' ' + c.description).toLowerCase().includes(q.toLowerCase()))
  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-extrabold">All courses</h1>
          <p className="text-slate-400 mt-1">Every course ends with an exam · assignments · real projects.</p>
        </div>
        <input className="input max-w-xs" placeholder="Search courses…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((c) => (
          <Link key={c.slug} to={`/courses/${c.slug}`} className="card hover:border-accent transition">
            <div className="flex items-start justify-between">
              <div className="font-bold">{c.title}</div>
              <span className="badge bg-night-700 text-slate-400">{c.order}</span>
            </div>
            <p className="text-sm text-slate-400 mt-2">{c.description}</p>
            <div className="mt-4 flex gap-2 text-xs">
              <span className="badge bg-night-700 text-slate-300">{c.lessons} lessons</span>
              <span className="badge bg-night-700 text-slate-300">{c.exams} exam</span>
              <span className="badge bg-night-700 text-slate-300">{c.projects} projects</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
