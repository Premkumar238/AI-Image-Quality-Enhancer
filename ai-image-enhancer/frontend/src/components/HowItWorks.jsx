const STEPS = [
  {
    title: 'Upload',
    text: 'Drop a JPG or PNG photo — blurry, small, or low quality works best.',
  },
  {
    title: 'Choose scale',
    text: 'Pick 2x, 3x, or 4x. Start with 2x for speed, use 4x for maximum detail.',
  },
  {
    title: 'Convert',
    text: 'Real-ESRGAN upscales and restores detail. CPU mode may take up to a minute.',
  },
  {
    title: 'Download',
    text: 'Save the enhanced result and compare it side by side with the original.',
  },
]

function HowItWorks() {
  return (
    <section id="how-it-works" className="scroll-mt-24">
      <div className="mb-8 max-w-2xl">
        <h2 className="text-2xl font-bold text-slate-900 sm:text-3xl">How it works</h2>
        <p className="mt-2 text-slate-600">
          Four simple steps from a soft photo to a sharper, larger result.
        </p>
      </div>

      <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map((step, index) => (
          <li
            key={step.title}
            className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
          >
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-indigo-600 text-sm font-bold text-white">
              {index + 1}
            </span>
            <h3 className="mt-4 text-base font-semibold text-slate-900">{step.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-slate-600">{step.text}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}

export default HowItWorks
