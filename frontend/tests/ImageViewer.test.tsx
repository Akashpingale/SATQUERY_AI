import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { ImageViewer } from '@/components/viewer/ImageViewer'
import type { RasterAsset } from '@/lib/api'

import { georeferencedAsset } from './fixtures'

/**
 * Regression tests for the preview-collapse bug.
 *
 * A 512x512 raster rendered as a thin horizontal strip because the minimum height sat on an
 * absolutely positioned wrapper (out of flow => contributes no height), so its `flex-1`
 * parent collapsed to ~0px inside the scrolling panel body. jsdom does not compute layout,
 * so these tests assert the CSS contract that guarantees a non-zero, aspect-preserving box.
 */

/** A 512x512 SAR raster shaped like the real upload (3 bands, uint8, PNG preview). */
const sarAsset: RasterAsset = {
  ...georeferencedAsset,
  id: 'asset-sar-512',
  original_filename: '00020.png',
  driver: 'PNG',
  modality: 'sar',
  detected_modality: 'sar',
  width: 512,
  height: 512,
  band_count: 3,
  dtype: 'uint8',
  preview_url: '/api/v1/uploads/asset-sar-512/preview',
}

/** A deliberately non-square raster, to prove the ratio is not forced to a square. */
const wideAsset: RasterAsset = {
  ...georeferencedAsset,
  id: 'asset-wide',
  original_filename: 'wide_scene.tif',
  width: 1024,
  height: 256,
}

function previewImage(name: string) {
  return screen.getByRole('img', { name: new RegExp(`Preview of ${name}`) })
}

/** The positioned, in-flow box that owns the preview's height. */
function viewportBox(img: HTMLElement): HTMLElement {
  // img -> absolute centering wrapper -> in-flow container (relative, flex-1, min-h)
  const wrapper = img.parentElement as HTMLElement
  return wrapper.parentElement as HTMLElement
}

describe('ImageViewer preview rendering', () => {
  it('renders a 512x512 SAR preview inside a box with a non-zero minimum height', () => {
    render(<ImageViewer asset={sarAsset} />)

    const img = previewImage('00020.png')
    const box = viewportBox(img)

    // The in-flow container must carry a min-height, otherwise flex-basis 0 collapses it.
    expect(box.className).toMatch(/min-h-\[\d+(\.\d+)?rem\]/)
    expect(box.className).toContain('relative')
    expect(box.className).toContain('flex-1')
  })

  it('does NOT put the minimum height on the out-of-flow wrapper (the original bug)', () => {
    render(<ImageViewer asset={sarAsset} />)

    const wrapper = previewImage('00020.png').parentElement as HTMLElement
    expect(wrapper.className).toContain('absolute')
    // An absolutely positioned element cannot give its parent height, so a min-h here is
    // exactly what caused the collapsed strip.
    expect(wrapper.className).not.toMatch(/min-h-/)
  })

  it('preserves the source aspect ratio without cropping', () => {
    render(<ImageViewer asset={sarAsset} />)

    const img = previewImage('00020.png')
    // object-contain letterboxes (never crops); both max bounds keep it inside the box.
    expect(img.className).toContain('object-contain')
    expect(img.className).toContain('max-h-full')
    expect(img.className).toContain('max-w-full')
    expect(img.className).not.toContain('object-cover')
  })

  it('uses identical rendering logic for optical and SAR previews', () => {
    const { unmount } = render(<ImageViewer asset={sarAsset} />)
    const sarClasses = previewImage('00020.png').className
    const sarBox = viewportBox(previewImage('00020.png')).className
    unmount()

    render(<ImageViewer asset={georeferencedAsset} />)
    const opticalClasses = previewImage('scene_optical.tif').className
    const opticalBox = viewportBox(previewImage('scene_optical.tif')).className

    expect(opticalClasses).toBe(sarClasses)
    expect(opticalBox).toBe(sarBox)
  })

  it('does not force a square box for non-square rasters', () => {
    render(<ImageViewer asset={wideAsset} />)
    const img = previewImage('wide_scene.tif')
    // No fixed aspect utility is applied; object-contain derives the ratio from the source.
    expect(img.className).toContain('object-contain')
    expect(viewportBox(img).className).not.toMatch(/aspect-square/)
  })

  it('keeps the preview box intact while zooming in and out', async () => {
    render(<ImageViewer asset={sarAsset} />)

    const before = viewportBox(previewImage('00020.png')).className
    expect(screen.getByText('100%')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(screen.getByText('125%')).toBeInTheDocument()

    // Zoom is a CSS transform on the image; the container must not lose its height floor.
    const zoomedBox = viewportBox(previewImage('00020.png'))
    expect(zoomedBox.className).toBe(before)
    expect(zoomedBox.className).toMatch(/min-h-\[\d+(\.\d+)?rem\]/)
    expect(previewImage('00020.png').style.transform).toContain('scale(1.25)')

    await userEvent.click(screen.getByRole('button', { name: 'Zoom out' }))
    expect(screen.getByText('100%')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Fit to screen' }))
    expect(screen.getByText('100%')).toBeInTheDocument()
    expect(viewportBox(previewImage('00020.png')).className).toBe(before)
  })

  it('still reports missing imagery instead of an empty box', () => {
    render(<ImageViewer asset={null} />)
    expect(screen.getByText('No imagery selected')).toBeInTheDocument()
  })
})
