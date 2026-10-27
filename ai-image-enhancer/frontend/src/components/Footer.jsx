function Footer() {
  return (
    <footer id="contact" className="scroll-mt-24 border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col items-center gap-6 px-4 py-10 text-center sm:px-6 lg:px-8">
        <div>
          <p className="text-sm text-slate-500">Powered by</p>
          <p className="mt-1 text-base font-semibold text-slate-900">Prem Kumar</p>
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

        <div className="flex flex-wrap justify-center gap-4 border-t border-slate-100 pt-4 text-sm text-slate-500">
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
