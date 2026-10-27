function Footer() {
  return (
    <footer id="contact" className="scroll-mt-24 border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-5 px-4 py-8 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-sm font-semibold text-slate-900">
              Powered by Prem Kumar
            </p>
            <p className="mt-1 text-sm text-slate-500">
              AI Image Quality Enhancer
            </p>
          </div>

          <div className="text-sm text-slate-600">
            <p className="font-medium text-slate-800">Contact</p>
            <a
              href="mailto:premkumarr238@gmail.com"
              className="mt-1 block hover:text-indigo-600"
            >
              premkumarr238@gmail.com
            </a>
            <a
              href="mailto:prem@devitytechnologies.com"
              className="mt-1 block hover:text-indigo-600"
            >
              prem@devitytechnologies.com
            </a>
          </div>
        </div>

        <div className="flex flex-wrap gap-4 border-t border-slate-100 pt-4 text-sm text-slate-500">
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
