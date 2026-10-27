function Header() {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-white/90 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
        <a href="#top" className="flex items-center gap-3 no-underline">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-sm font-bold text-white">
            AI
          </span>
          <span>
            <span className="block text-base font-semibold text-slate-900 sm:text-lg">
              AI Image Quality Enhancer
            </span>
            <span className="hidden text-xs text-slate-500 sm:block">
              Powered by Prem Kumar
            </span>
          </span>
        </a>

        <nav className="hidden items-center gap-6 text-sm font-medium text-slate-600 md:flex">
          <a href="#enhance" className="hover:text-indigo-600">
            Enhance
          </a>
          <a href="#how-it-works" className="hover:text-indigo-600">
            How it works
          </a>
          <a href="#features" className="hover:text-indigo-600">
            Features
          </a>
          <a href="#contact" className="hover:text-indigo-600">
            Contact
          </a>
        </nav>

        <p className="hidden text-right text-xs text-slate-500 sm:block md:text-sm">
          Powered by
          <span className="mt-0.5 block font-semibold text-slate-800">Prem Kumar</span>
        </p>
      </div>
    </header>
  )
}

export default Header
