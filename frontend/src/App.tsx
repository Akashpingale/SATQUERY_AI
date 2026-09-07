import { Route, Routes } from 'react-router-dom'

import { NotFound } from '@/pages/NotFound'
import { Workspace } from '@/pages/Workspace'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Workspace />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
