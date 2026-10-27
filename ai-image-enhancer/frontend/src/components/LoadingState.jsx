function LoadingState({ message = 'Processing your image...', detail }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-8 text-center">
      <div
        className="h-10 w-10 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600"
        role="status"
        aria-label="Loading"
      />
      <p className="text-sm font-medium text-slate-700">{message}</p>
      {detail ? <p className="max-w-xs text-xs text-slate-500">{detail}</p> : null}
    </div>
  )
}

export default LoadingState
