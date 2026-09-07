import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { SystemStatus } from '@/components/SystemStatus'

describe('SystemStatus', () => {
  it('renders an accessible online status with a visible text label', () => {
    render(<SystemStatus status="online" />)
    const status = screen.getByRole('status')
    // Conveyed by text (not color alone).
    expect(status).toHaveTextContent('Online')
  })

  it('renders offline state', () => {
    render(<SystemStatus status="offline" />)
    expect(screen.getByRole('status')).toHaveTextContent('Offline')
  })

  it('renders checking state', () => {
    render(<SystemStatus status="checking" />)
    expect(screen.getByRole('status')).toHaveTextContent('Checking')
  })
})
