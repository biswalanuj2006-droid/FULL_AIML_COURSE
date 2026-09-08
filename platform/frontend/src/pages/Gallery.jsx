import React from 'react'

export default function Gallery() {
  return (
    <div>
      <h1 className="text-3xl font-extrabold mb-2">Diagram gallery</h1>
      <p className="text-slate-400 mb-6">All 89 generated teaching panels — thumbnails, captions, dimensions.</p>
      <div className="card !p-2">
        <iframe
          src="/assets/diagrams/gallery/index.html"
          title="Diagram gallery"
          className="w-full h-[75vh] rounded-xl bg-white"
        />
      </div>
    </div>
  )
}
