import React, { useRef } from "react";
import "./Home.css";

const authors = [
  {
    name: "Aryan Das",
    email: "aryandas156@gmail.com",
    scholar: "https://scholar.google.com/citations?user=D_B6X9gAAAAJ&hl=en",
    linkedin: "https://www.linkedin.com/in/aryan--das/",
    github: "https://github.com/arya-domain",
    affiliation: "Dept. of CSE, VIT Bhopal",
  },
  {
    name: "Tanishq Rachamalla",
    email: "tanishqrachamalla12@gmail.com",
    scholar: "https://scholar.google.com/citations?user=240vcAsAAAAJ&hl=en",
    linkedin: "https://in.linkedin.com/in/tanishq-rachamalla-5a759b234",
    github: "https://github.com/tanishq251",
    affiliation: "Dept. of IT, SAHE, Andhra Pradesh",
  },
  {
    name: "Pravendra Singh",
    email: "pravendra.singh@cs.iitr.ac.in",
    scholar: "https://scholar.google.com/citations?user=YwDTxJMAAAAJ&hl=en",
    linkedin: "https://in.linkedin.com/in/pravendra-singh-001",
    website: "https://sites.google.com/view/pravendra/",
    affiliation: "Dept. of CSE, IIT Roorkee",
  },
  {
    name: "Koushik Biswas",
    email: "koushikb@iiitd.ac.in",
    dblp: "https://dblp.org/pid/274/2151.html",
    linkedin: "https://www.linkedin.com/in/koushik-biswas-ml",
    affiliation: "Dept. of CSE, IIIT Delhi",
  },
  {
    name: "Vinay Kumar Verma",
    email: "vinayugc@gmail.com",
    scholar: "https://scholar.google.com/citations?user=7x6GZ1EAAAAJ&hl=en",
    linkedin: "https://in.linkedin.com/in/vinay-kumar-verma-6a315468",
    github: "https://github.com/vkverma01",
    affiliation: "Research Scientist, Amazon India",
  },
  {
    name: "Swalpa Kumar Roy",
    email: "swalpa@agemc.ac.in",
    scholar: "https://scholar.google.com/citations?user=1WVrFGwAAAAJ&hl=en",
    linkedin: "https://in.linkedin.com/in/swalpa-kumar-roy-9b51234a",
    website: "https://swalpa.github.io/",
    affiliation: "Dept. of CSE, Alipurduar Govt. Engg. & Mgmt. College",
  },
];

const Home = () => {
  const bibtexEntry = `@misc{das2025hyperspectralimagelandcover,
      title={Hyperspectral Image Land Cover Captioning Dataset for Vision Language Models},
      author={Aryan Das and Tanishq Rachamalla and Pravendra Singh and Koushik Biswas and Vinay Kumar Verma and Swalpa Kumar Roy},
      year={2025},
      eprint={2505.12217},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.12217},
  }`;

  const bibtexSectionRef = useRef(null); //

  const scrollToBibtexSection = () => {
    bibtexSectionRef.current.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <>
      {/* Authors Section */}
      <section className="main-section">
        <h1>
          Hyperspectral Imaging Land Cover Captioning for Vision Language Models
        </h1>
        <h2 className="arxiv">
          <a href="https://arxiv.org/abs/2505.12217">arXiv 2025</a>
        </h2>

        <div className="project-buttons">
          <a
            href="https://arxiv.org/pdf/2505.12217"
            className="project-btn"
            target="_blank"
            rel="noreferrer"
          >
            Paper
          </a>
          <a
            href="https://github.com/arya-domain/HyperCap"
            className="project-btn"
            target="_blank"
            rel="noreferrer"
          >
            Code
          </a>
          <a className="project-btn" onClick={scrollToBibtexSection}>
            BibTeX
          </a>
        </div>

        <h3>Authors</h3>
        <div className="author-line">
          {authors.map((author, index) => (
            <div className="author-hovercard" key={index}>
              <span className="author-name">{author.name}</span>

              <div className="hovercard">
                {/* <p><strong>{author.name}</strong></p> */}
                <p>{author.affiliation}</p>
                {author.email && (
                  <p>
                    <a href={`mailto:${author.email}`}>{author.email}</a>
                  </p>
                )}
                {author.scholar && (
                  <p>
                    <a href={author.scholar} target="_blank" rel="noreferrer">
                      Google Scholar
                    </a>
                  </p>
                )}
                {author.dblp && (
                  <p>
                    <a href={author.dblp} target="_blank" rel="noreferrer">
                      DBLP
                    </a>
                  </p>
                )}
                {author.linkedin && (
                  <p>
                    <a href={author.linkedin} target="_blank" rel="noreferrer">
                      LinkedIn
                    </a>
                  </p>
                )}
                {author.github && (
                  <p>
                    <a href={author.github} target="_blank" rel="noreferrer">
                      GitHub
                    </a>
                  </p>
                )}
                {author.website && (
                  <p>
                    <a href={author.website} target="_blank" rel="noreferrer">
                      Website
                    </a>
                  </p>
                )}
              </div>
              {index !== authors.length - 1 && <span className="dot">·</span>}
            </div>
          ))}
        </div>
      </section>

      {/* Abstract Section */}
      <section className="abstract-section">
        <h2>Abstract</h2>
        <p className="abstract-text">
          We introduce HyperCap, the first large-scale hyperspectral captioning
          dataset designed to enhance model performance and effectiveness in
          remote sensing applications. Unlike traditional hyperspectral imaging
          (HSI) datasets that focus solely on classification tasks, HyperCap
          integrates spectral data with pixel-wise textual annotations, enabling
          deeper semantic understanding of hyperspectral imagery. This dataset
          enhances model performance in tasks like classification and feature
          extraction, providing a valuable resource for advanced remote sensing
          applications. HyperCap is constructed from four benchmark datasets and
          annotated through a hybrid approach combining automated and manual
          methods to ensure accuracy and consistency. Empirical evaluations
          using state-of-the-art encoders and diverse fusion techniques
          demonstrate significant improvements in classification performance.
          These results underscore the potential of vision-language learning in
          HSI and position HyperCap as a foundational dataset for future
          research in the field.
        </p>
      </section>

      {/* Highlights Section */}
      <section className="highlights-section">
        <h2>Highlights</h2>
        <ul className="highlight-list">
          <li>
            We propose <strong>HyperCap</strong>, the first large-scale HSI
            captioning dataset for Remote Sensing, providing fine-grained,
            pixel-wise textual descriptions for HSI images.
          </li>
          <li>
            {" "}
            Unlike traditional HSI datasets that focus solely on classification,
            HyperCap combines spectral data with textual annotations. This
            integration allows models to generate human-readable explanations,
            thereby enhancing semantic understanding.
          </li>
          <li>
            We evaluate the effectiveness of existing methods on HyperCap,
            establishing a foundation for future research in vision-language
            learning for HSI imaging.
          </li>
        </ul>
      </section>

      <section
        id="bibtex-section"
        ref={bibtexSectionRef}
        className="bibtex-section"
      >
        <h2>BibTeX</h2>
        <pre className="bibtex-code">
          <code>{bibtexEntry}</code>
        </pre>
      </section>
    </>
  );
};

export default Home;
