function Header({ statusIndicator }) {
  return (
    <header className="border-b border-gray-200 bg-white px-6 py-4">
      <div className="mx-auto flex max-w-4xl items-center justify-between gap-4">
        <h1 className="text-2xl font-bold text-gray-900">AI Image Enhancer</h1>
        {statusIndicator}
      </div>
    </header>
  )
}

export default Header
