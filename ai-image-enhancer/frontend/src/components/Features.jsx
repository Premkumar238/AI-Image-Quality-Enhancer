const FEATURES = [
  {
    title: 'AI upscaling',
    text: 'Real-ESRGAN enlarges photos while recovering edges and texture instead of just stretching pixels.',
  },
  {
    title: 'Sharper details',
    text: 'Reduce softness and noise so faces, text, and landscapes look clearer after conversion.',
  },
  {
    title: 'Side-by-side compare',
    text: 'See original and enhanced sizes together, then download the improved image instantly.',
  },
  {
    title: 'Local processing',
    text: 'Your images are processed on your machine through the local backend — no cloud upload required.',
  },
]

function Features() {
  return (
    <section id="features" className="scroll-mt-24">
      <div className="mb-8 max-w-2xl">
        <h2 className="text-2xl font-bold text-slate-900 sm:text-3xl">Features</h2>
        <p className="mt-2 text-slate-600">
          Built for practical photo enhancement with a clear before-and-after workflow.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        {FEATURES.map((feature) => (
          <article
            key={feature.title}
            className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
          >
            <h3 className="text-base font-semibold text-slate-900">{feature.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">{feature.text}</p>
          </article>
        ))}
      </div>
    </section>
  )
}

export default Features
