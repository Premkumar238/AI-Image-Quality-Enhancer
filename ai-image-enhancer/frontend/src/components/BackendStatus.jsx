function BackendStatus({ status, opencvAvailable, pipeline, realesrganLoaded }) {
  if (status === 'connected') {
    if (realesrganLoaded || pipeline === 'realesrgan-x4plus') {
      return (
        <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700">
          Real-ESRGAN Ready
        </span>
      )
    }

    if (opencvAvailable === false) {
      return (
        <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-800">
          Backend needs setup
        </span>
      )
    }

    return (
      <span className="rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-800">
        Fallback mode
      </span>
    )
  }

  if (status === 'offline') {
    return (
      <span className="rounded-full bg-red-100 px-3 py-1 text-xs font-medium text-red-700">
        Backend Offline
      </span>
    )
  }

  return (
    <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-500">
      Checking backend...
    </span>
  )
}

export default BackendStatus
