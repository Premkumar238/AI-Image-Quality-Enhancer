function LoadingState({ message = 'Processing your image...' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-8">
      <div
        className="h-10 w-10 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600"
        role="status"
        aria-label="Loading"
      />
      <p className="text-sm text-gray-600">{message}</p>
    </div>
  )
}

export default LoadingState
