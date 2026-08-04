const DocsPage = () => {
  return (
    <div className="p-6 lg:p-8 max-w-3xl space-y-8">
      <header className="animate-fade-in-up">
        <h1 className="text-2xl font-display font-bold text-foreground">Quick Help</h1>
        <p className="text-sm text-muted-foreground mt-1">Understanding KNF descriptors and analysis workflow</p>
      </header>

      <DocSection title="What are KNF Descriptors?">
        <p>
          KNF descriptors are a set of nine molecular features (<code className="text-primary font-mono text-sm">f1</code>–<code className="text-primary font-mono text-sm">f9</code>) 
          derived from Non-Covalent Interaction (NCI) analysis. They capture the spatial distribution and 
          characteristics of intermolecular forces within a molecular system. KNF Studio computes these 
          descriptors in batch and derives two aggregate metrics: <strong>SNCI</strong> and <strong>SCDI</strong>.
        </p>
      </DocSection>

      <DocSection title="SNCI – Summed NCI Index">
        <p>
          The SNCI is a scalar aggregation of attractive non-covalent interaction regions. 
          It summarizes the total NCI contribution across the molecular volume. Higher absolute values 
          indicate stronger or more distributed non-covalent interactions.
        </p>
      </DocSection>

      <DocSection title="SCDI – Summed Charge Density Index">
        <p>
          <code className="text-primary font-mono text-sm">SCDI_variance</code> is the area-weighted variance
          of the xTB COSMO surface charge. It measures charge heterogeneity while retaining the raw physical scale.
        </p>
      </DocSection>

      <DocSection title="Normalization: SNCI_Norm & SCDI_Norm">
        <p>
          SNCI and raw SCDI variance are normalized to <code className="font-mono text-primary text-sm">[0, 1]</code> within
          each batch. SCDI uses inverse min-max normalization, so lower variance maps to a higher normalized value.
        </p>
        <div className="mt-3 rounded-lg bg-muted p-4 font-mono text-xs text-muted-foreground">
          SNCI_Norm = (SNCI - min(SNCI)) / (max(SNCI) - min(SNCI))<br />
          SCDI_Norm = 1 - (SCDI_variance - min) / (max - min)
        </div>
      </DocSection>

      <DocSection title="Quadrant Analysis">
        <p>
          Molecules are classified into four quadrants based on their normalized SNCI and SCDI values 
          relative to the <strong>batch median</strong>:
        </p>
        <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-lg bg-card border border-border p-3">
            <p className="font-semibold text-foreground">Q1 (High SNCI, High SCDI)</p>
            <p className="text-xs text-muted-foreground mt-1">Strong NCI + high charge density</p>
          </div>
          <div className="rounded-lg bg-card border border-border p-3">
            <p className="font-semibold text-foreground">Q2 (Low SNCI, High SCDI)</p>
            <p className="text-xs text-muted-foreground mt-1">Weak NCI + high charge density</p>
          </div>
          <div className="rounded-lg bg-card border border-border p-3">
            <p className="font-semibold text-foreground">Q3 (Low SNCI, Low SCDI)</p>
            <p className="text-xs text-muted-foreground mt-1">Weak NCI + low charge density</p>
          </div>
          <div className="rounded-lg bg-card border border-border p-3">
            <p className="font-semibold text-foreground">Q4 (High SNCI, Low SCDI)</p>
            <p className="text-xs text-muted-foreground mt-1">Strong NCI + low charge density</p>
          </div>
        </div>
      </DocSection>

      <DocSection title="Stop Mid-Way Behavior">
        <p>
          You can stop a running pipeline at any time. When stopped:
        </p>
        <ul className="list-disc list-inside text-muted-foreground space-y-1 mt-2 text-sm">
          <li>Currently running tasks will finish gracefully</li>
          <li>Queued tasks will be marked as <code className="font-mono text-warning text-xs">stopped</code></li>
          <li>All completed results remain available for analysis and export</li>
          <li>Normalization and quadrant analysis are recomputed on partial results</li>
        </ul>
      </DocSection>
    </div>
  );
};

function DocSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-lg font-display font-semibold text-foreground">{title}</h2>
      <div className="text-sm text-muted-foreground leading-relaxed">{children}</div>
    </section>
  );
}

export default DocsPage;
