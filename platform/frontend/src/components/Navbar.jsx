import React from 'react'
import { Link, NavLink } from 'react-router-dom'
import { useAuth } from '../auth.jsx'
import { levelInfo } from '../lib/api.js'

export default function Navbar() {
  const { user, logout } = useAuth()
  const link = ({ isActive }) =>
    `px-3 py-2 rounded-lg text-sm font-medium ${isActive ? 'text-accent bg-night-700' : 'text-slate-300 hover:text-white'}`
  return (
    <header className="sticky top-0 z-40 bg-night-900/90 backdrop-blur border-b border-night-600">
      <div className="max-w-6xl mx-auto px-4 h-16 flex items-center gap-2">
        <Link to="/" className="font-extrabold text-lg tracking-tight mr-4">
          <span className="text-accent">AI/ML</span> Academy
        </Link>
        <nav className="hidden md:flex items-center gap-1">
          <NavLink to="/courses" className={link}>Courses</NavLink>
          <NavLink to="/leaderboard" className={link}>Leaderboard</NavLink>
          <NavLink to="/gallery" className={link}>Gallery</NavLink>
          {user?.role === 'admin' && <NavLink to="/admin" className={link}>Admin</NavLink>}
        </nav>
        <div className="ml-auto flex items-center gap-3">
          {user ? (
            <>
              <Link to="/dashboard" className={`badge border ${levelInfo(user.level).ring}`}>
                {levelInfo(user.level).label} · {user.xp} XP
              </Link>
              <button onClick={logout} className="btn-ghost !py-1.5">Logout</button>
            </>
          ) : (
            <Link to="/login" className="btn !py-1.5">Sign in</Link>
          )}
        </div>
      </div>
    </header>
  )
}
