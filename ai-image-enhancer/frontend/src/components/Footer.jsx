function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <p>AI Image Quality Enhancer · Real-ESRGAN powered</p>
        <div className="flex gap-4">
          <a href="#enhance" className="hover:text-indigo-600">
            Enhance
          </a>
          <a href="#how-it-works" className="hover:text-indigo-600">
            How it works
          </a>
          <a href="#features" className="hover:text-indigo-600">
            Features
          </a>
        </div>
      </div>
    </footer>
  )
}

export default Footer
