import React from 'react'
import useBaseUrl from '@docusaurus/useBaseUrl'
import {useColorMode} from '@docusaurus/theme-common';

export default function Index({src, darkSrc, caption, width, height}) {
    const isDarkTheme = useColorMode().colorMode === "dark";
    const darkImageSrc = darkSrc || src;
    width = width || "auto";
    height = height || "auto";
    return (
        <figure style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
            <img src={isDarkTheme ? useBaseUrl(darkImageSrc) : useBaseUrl(src)} alt={caption} width={width} height={height}/>
            { caption &&
            <figcaption style={{textAlign: 'center', fontWeight: 300, marginTop: '0.5rem'}}>{`Figure: ${caption}`}</figcaption>
            }
        </figure>
    )
}
