import '@testing-library/jest-dom/vitest';

class ResizeObserverStub {
  observe() {
    // noop
  }
  unobserve() {
    // noop
  }
  disconnect() {
    // noop
  }
}

if (typeof window !== 'undefined' && !('ResizeObserver' in window)) {
  // @ts-expect-error - assigning to global
  window.ResizeObserver = ResizeObserverStub;
}
