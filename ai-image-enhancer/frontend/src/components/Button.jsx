function Button({
  children,
  onClick,
  type = 'button',
  disabled = false,
  variant = 'primary',
  className = '',
}) {
  const variants = {
    primary:
      'bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-indigo-400',
    secondary:
      'border border-slate-300 bg-white text-slate-800 hover:bg-slate-50 disabled:text-slate-400',
    ghost:
      'bg-transparent text-slate-600 hover:bg-slate-100 disabled:text-slate-400',
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center rounded-xl px-4 py-2.5 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-60 ${variants[variant] || variants.primary} ${className}`}
    >
      {children}
    </button>
  )
}

export default Button
