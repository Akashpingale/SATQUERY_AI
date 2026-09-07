import { useCallback, useState } from 'react'

import {
  ApiError,
  type CompatibilityPurpose,
  type CompatibilityResponse,
  type Modality,
  type RasterAsset,
  type UploadRole,
  api,
} from '@/lib/api'

/** Per-slot upload state, including the error path. */
export interface SlotState {
  status: 'empty' | 'uploading' | 'ready' | 'error'
  asset?: RasterAsset
  error?: string
  issues?: { code: string; message: string; severity: string }[]
}

export type SlotMap = Record<UploadRole, SlotState>

const EMPTY: SlotState = { status: 'empty' }

const INITIAL_SLOTS: SlotMap = {
  single: EMPTY,
  optical: EMPTY,
  sar: EMPTY,
  t1: EMPTY,
  t2: EMPTY,
}

/** Modality implied by a slot; 'single' leaves detection to the backend. */
function modalityForRole(role: UploadRole): Modality {
  if (role === 'optical' || role === 't1' || role === 't2') return 'optical'
  if (role === 'sar') return 'sar'
  return 'unknown'
}

/**
 * Manages uploads for the four input slots and pair-compatibility checks.
 *
 * Compatibility is requested from the backend - the UI never decides on its own
 * whether two images can be analyzed together.
 */
export function useUploads() {
  const [slots, setSlots] = useState<SlotMap>(INITIAL_SLOTS)
  const [compatibility, setCompatibility] = useState<CompatibilityResponse | null>(null)
  const [compatibilityError, setCompatibilityError] = useState<string | null>(null)

  const setSlot = useCallback((role: UploadRole, next: SlotState) => {
    setSlots((prev) => ({ ...prev, [role]: next }))
  }, [])

  const upload = useCallback(
    async (role: UploadRole, file: File) => {
      setSlot(role, { status: 'uploading' })
      setCompatibility(null)
      setCompatibilityError(null)
      try {
        const asset = await api.uploadRaster(file, role, modalityForRole(role))
        setSlot(role, { status: 'ready', asset })
      } catch (error) {
        const apiError = error instanceof ApiError ? error : null
        setSlot(role, {
          status: 'error',
          error: apiError?.message ?? 'Upload failed. Please try again.',
          issues: apiError?.issues,
        })
      }
    },
    [setSlot],
  )

  const remove = useCallback(
    async (role: UploadRole) => {
      const current = slots[role]
      setSlot(role, EMPTY)
      setCompatibility(null)
      setCompatibilityError(null)
      if (current.asset) {
        try {
          await api.deleteUpload(current.asset.id)
        } catch {
          // Removing the slot locally is the user-visible action; a failed server
          // delete is not worth interrupting them for.
        }
      }
    },
    [slots, setSlot],
  )

  const checkPair = useCallback(
    async (roleA: UploadRole, roleB: UploadRole, purpose: CompatibilityPurpose) => {
      const a = slots[roleA]?.asset
      const b = slots[roleB]?.asset
      if (!a || !b) return
      setCompatibility(null)
      setCompatibilityError(null)
      try {
        setCompatibility(await api.checkCompatibility(a.id, b.id, purpose))
      } catch (error) {
        setCompatibilityError(
          error instanceof ApiError ? error.message : 'Compatibility check failed.',
        )
      }
    },
    [slots],
  )

  const assets = (Object.values(slots) as SlotState[])
    .map((slot) => slot.asset)
    .filter((asset): asset is RasterAsset => Boolean(asset))

  return { slots, assets, upload, remove, checkPair, compatibility, compatibilityError }
}
