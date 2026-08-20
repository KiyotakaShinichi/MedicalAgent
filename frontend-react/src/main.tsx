import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ToastProvider } from './components/ui/Toast'
import { ErrorBoundary } from './components/ui/ErrorBoundary'
import { installGlobalErrorHandlers, reportError } from './lib/telemetry'

// Catch failures that never reach a React boundary: unhandled promise
// rejections and top-level script errors. Installed before render so a crash
// during the first paint is still reported.
installGlobalErrorHandlers()

const container = document.getElementById('root')
if (!container) {
  // Nothing can render, so make the reason explicit instead of throwing an
  // opaque "null is not an object" from a non-null assertion.
  reportError(new Error('Root container #root is missing from the document'), {
    surface: 'main.bootstrap',
  })
  throw new Error('Unable to start NLCare: #root container not found')
}

createRoot(container).render(
  <StrictMode>
    <ErrorBoundary surface="the application">
      <ToastProvider>
        <App />
      </ToastProvider>
    </ErrorBoundary>
  </StrictMode>,
)
