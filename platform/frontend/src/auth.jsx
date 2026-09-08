import React, { createContext, useContext, useEffect, useState } from 'react'
import { api, getToken, setToken } from './lib/api.js'

const AuthCtx = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [ready, setReady] = useState(false)

  const refresh = async () => {
    if (!getToken()) { setUser(null); setReady(true); return }
    try { setUser(await api('/api/auth/me')) }
    catch { setToken(''); setUser(null) }
    setReady(true)
  }

  useEffect(() => { refresh() }, [])

  const login = async (email, password) => {
    const d = await api('/api/auth/login', { method: 'POST', body: { email, password } })
    setToken(d.token); setUser(d.user); return d.user
  }

  const register = async (email, name, password) => {
    await api('/api/auth/register', { method: 'POST', body: { email, name, password } })
    return login(email, password)
  }

  const logout = () => { setToken(''); setUser(null) }

  return (
    <AuthCtx.Provider value={{ user, ready, login, register, logout, refresh }}>
      {children}
    </AuthCtx.Provider>
  )
}

export function useAuth() {
  return useContext(AuthCtx)
}
