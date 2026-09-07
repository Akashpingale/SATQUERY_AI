/**
 * Lightweight 3D "mission control" centerpiece for the empty workspace state.
 *
 * A low-poly wireframe globe with an orbiting satellite marker and a faint atmosphere
 * glow. Deliberately cheap: icosahedron wireframe (no textures), slow rotation, capped
 * device-pixel-ratio, and it stops animating when the tab is hidden or the user prefers
 * reduced motion. Rendered only as a background layer of the empty state, so it never
 * covers imagery, controls, or results.
 *
 * This file is lazy-loaded and only mounted when WebGL is available (see EmptyWorkspace),
 * so it is excluded from the initial bundle and from jsdom test runs.
 */
import { useMemo, useRef } from 'react'

import { Canvas, useFrame } from '@react-three/fiber'
import type { Group, Mesh } from 'three'

function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches === true
  )
}

function Globe() {
  const globe = useRef<Mesh>(null)
  const orbit = useRef<Group>(null)
  const reduce = useMemo(prefersReducedMotion, [])

  useFrame((_state, delta) => {
    if (reduce) return
    if (globe.current) globe.current.rotation.y += delta * 0.12
    if (orbit.current) orbit.current.rotation.y += delta * 0.35
  })

  return (
    <group>
      {/* Wireframe planet */}
      <mesh ref={globe}>
        <icosahedronGeometry args={[1.6, 4]} />
        <meshStandardMaterial
          color="#0b3b52"
          emissive="#0a2740"
          wireframe
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Solid inner sphere for subtle depth */}
      <mesh scale={0.985}>
        <icosahedronGeometry args={[1.6, 3]} />
        <meshStandardMaterial color="#071726" emissive="#04121f" roughness={1} metalness={0} />
      </mesh>

      {/* Faint atmosphere shell */}
      <mesh scale={1.12}>
        <sphereGeometry args={[1.6, 32, 32]} />
        <meshBasicMaterial color="#22d3ee" transparent opacity={0.05} />
      </mesh>

      {/* Orbit ring + satellite marker */}
      <group ref={orbit} rotation={[Math.PI / 2.6, 0, 0]}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[2.5, 0.006, 8, 128]} />
          <meshBasicMaterial color="#38bdf8" transparent opacity={0.4} />
        </mesh>
        <mesh position={[2.5, 0, 0]}>
          <boxGeometry args={[0.12, 0.12, 0.12]} />
          <meshBasicMaterial color="#a78bfa" />
        </mesh>
      </group>
    </group>
  )
}

export default function Globe3D() {
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 42 }}
      dpr={[1, 1.5]}
      gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
      style={{ pointerEvents: 'none' }}
      aria-hidden="true"
      frameloop="always"
    >
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 3, 5]} intensity={1.1} color="#7dd3fc" />
      <directionalLight position={[-4, -2, -3]} intensity={0.5} color="#a78bfa" />
      <Globe />
    </Canvas>
  )
}
