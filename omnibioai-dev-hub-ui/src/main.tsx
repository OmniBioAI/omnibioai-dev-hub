import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@omnibioai/design-tokens/tokens.css'
import './index.css'
import App from './App.tsx'
import { connectAppearanceMessages } from './embedAppearance'

connectAppearanceMessages()
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
