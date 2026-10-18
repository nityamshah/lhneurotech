import ChatInterface from './components/ChatInterface.jsx'

// Slice 1: hardcoded score. Slice 3 replaces this with useEngagementStream.
const HARDCODED_SCORE = 0.72

export default function App() {
  return <ChatInterface score={HARDCODED_SCORE} />
}
