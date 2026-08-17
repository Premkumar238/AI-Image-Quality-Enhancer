function BackendStatus({ status }) {
  if (status === 'connected') {
    return (
      <span className="rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700">
        Backend Connected
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
