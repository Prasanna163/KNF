/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useRef } from 'react';
import * as $3Dmol from '3dmol';

interface MoleculeViewerProps {
  data: string;
  format?: string;
  style?: any;
}

export const MoleculeViewer: React.FC<MoleculeViewerProps> = ({
  data,
  format = 'xyz',
  style = { stick: {} }
}) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  const viewerInstance = useRef<any>(null);

  useEffect(() => {
    if (!viewerRef.current) return;

    if (!viewerInstance.current) {
        viewerInstance.current = $3Dmol.createViewer(viewerRef.current, {
            defaultcolors: $3Dmol.rasmolElementColors
        });
    }

    const viewer = viewerInstance.current;
    viewer.clear();
    viewer.addModel(data, format);
    viewer.setStyle({}, style);
    viewer.zoomTo();
    viewer.render();

  }, [data, format, style]);

  return (
    <div
      ref={viewerRef}
      style={{ width: '100%', height: '100%', minHeight: '300px', position: 'relative' }}
    />
  );
};
