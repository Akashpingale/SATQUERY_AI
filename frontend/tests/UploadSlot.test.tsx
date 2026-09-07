import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { UploadSlot } from '@/components/upload/UploadSlot'

import { georeferencedAsset } from './fixtures'

const baseProps = {
  role: 'optical' as const,
  label: 'Optical / Multispectral',
  hint: 'Single-image analysis',
  onSelect: vi.fn(),
  onRemove: vi.fn(),
}

describe('UploadSlot', () => {
  it('renders an empty, keyboard-reachable drop zone', () => {
    render(<UploadSlot {...baseProps} state={{ status: 'empty' }} />)

    expect(screen.getByRole('button', { name: /drop a file or click to browse/i })).toBeEnabled()
    expect(screen.getByText(/GeoTIFF, TIFF, PNG, or JPEG/)).toBeInTheDocument()
    expect(screen.getByLabelText('Optical / Multispectral file')).toBeInTheDocument()
  })

  it('shows a validating state while the upload is checked', () => {
    render(<UploadSlot {...baseProps} state={{ status: 'uploading' }} />)

    expect(screen.getByText('Checking')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /validating/i })).toBeDisabled()
  })

  it('shows metadata and actions once a file is accepted', () => {
    render(<UploadSlot {...baseProps} state={{ status: 'ready', asset: georeferencedAsset }} />)

    expect(screen.getByText('Valid')).toBeInTheDocument()
    expect(screen.getByText('scene_optical.tif')).toBeInTheDocument()
    expect(screen.getByText('512 x 512 px')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Replace' })).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: /remove optical/i }),
    ).toBeInTheDocument()
  })

  it('explains a rejected file in plain language via an alert', () => {
    render(
      <UploadSlot
        {...baseProps}
        state={{
          status: 'error',
          error: 'The file could not be used as a remote-sensing raster.',
          issues: [
            {
              code: 'unreadable_raster',
              message: 'The file could not be opened as an image. It may be corrupted.',
              severity: 'error',
            },
          ],
        }}
      />,
    )

    expect(screen.getByText('Rejected')).toBeInTheDocument()
    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('could not be used as a remote-sensing raster')
    expect(screen.getByText(/may be corrupted/)).toBeInTheDocument()
  })

  it('calls onRemove when the remove action is used', async () => {
    const onRemove = vi.fn()
    render(
      <UploadSlot
        {...baseProps}
        onRemove={onRemove}
        state={{ status: 'ready', asset: georeferencedAsset }}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: /remove optical/i }))
    expect(onRemove).toHaveBeenCalledWith('optical')
  })
})
