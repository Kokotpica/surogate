import React from 'react'
import Grid from "@mui/material/Grid2";
import Card from "@mui/material/Card";
import {Button, CardActions, CardMedia, Typography} from "@mui/material";
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward';
import CardContent from "@mui/material/CardContent";

const logos = [
    {
        image: "/img/logo-black.png",
        classes: "bg-white",
        png: "/img/logo-black.png"
    },
    {
        image: "/img/logo-white.png",
        classes: "bg-black-alpha-90",
        png: "/img/logo-white.png"
    },
    {
        image: "/img/logo-simple-black.png",
        classes: "bg-white",
        svg: "/img/logo-simple-black.svg",
        png: "/img/logo-simple-black.png"
    },
    {
        image: "/img/logo-simple-white.png",
        classes: "bg-black-alpha-90",
        svg: "/img/logo-simple-white.svg",
        png: "/img/logo-simple-white.png"
    },
];

export default function Index() {
    return (
        <Grid container spacing={4}>
            {logos.map((logo, index) => (
                <Grid key={index}>
                    <Card>
                        <CardContent className='flex justify-content-center'>
                            <CardMedia
                                sx={{maxHeight: 100, width: 'auto'}}
                                component="img"
                                image={logo.image}
                                alt="StateMesh Logo"
                                className={`p-2 ${logo.classes}`}
                            />
                        </CardContent>
                        <CardActions>
                            { logo.svg &&
                                <Button size='small' href={logo.svg} target='_blank'>
                                    <Typography className='text-sm flex align-items-center'><ArrowDownwardIcon/> SVG</Typography>
                                </Button>
                            }
                            { logo.png &&
                                <Button size='small' href={logo.png} target='_blank'>
                                    <Typography className='text-sm flex align-items-center'><ArrowDownwardIcon/> PNG</Typography>
                                </Button>
                            }
                        </CardActions>
                    </Card>
                </Grid>
            ))}
        </Grid>
    );
}
