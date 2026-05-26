close all

%location and orientation
x0=0;
y0=0;
theta=0;

%standard diviations %adjust to get different shapes
sigmaFront=2;
sigmaSide=4/3;
sigmaBack=1;

%plot size and res
plotsizeX=4;
plotsizeY=4;
resolution=1000;

%create canvas
x=linspace(x0-plotsizeX,x0+plotsizeX,resolution);
y=linspace(y0-plotsizeY,y0+plotsizeY,resolution);
cost=zeros(length(x),length(y));


% create proxemic zones of people
personOne       =makeProxemicZone(0,2,x,y,pi+pi/2,sigmaFront,sigmaSide,sigmaBack);
personTwo       =makeProxemicZone(-cos(pi/6)*2,-sin(pi/6)*2,x,y,2/3*pi-pi/2,sigmaFront,sigmaSide,sigmaBack);
personThree     =makeProxemicZone(cos(pi/6)*2,-sin(pi/6)*2,x,y,2*pi-2/3*pi-pi/2,sigmaFront,sigmaSide,sigmaBack);

% add individual costs to costmap
cost=max(cost,personOne);
cost=max(cost,personTwo);
cost=max(cost,personThree);

%plotting and exporting
surf(x,y,cost);

shading interp
hold on
axis off
pbaspect([1 1 1])
view(0,90);
ax=gca;
exportgraphics(ax,'InteractionZone.png','Resolution',500);


function cost = makeProxemicZone(x0,y0,x,y,theta,sigmaFront,sigmaSide,sigmaBack)
    social=zeros(length(x),length(y));
    passing=zeros(length(x),length(y));
    for i=1:length(x)
        for j=1:length(y)
            socialcost=asymmetricGaus(x(i),y(j),x0,y0,theta,sigmaFront,sigmaSide,sigmaBack);
            social(j,i)=classifyCost(socialcost);
            passingCost=asymmetricGaus(x(i),y(j),x0,y0,theta-pi/2,2,2/3,sigmaBack);%adjust the last three values to get a differently shaped passing layer
            passing(j,i)=classifyCost(passingCost);
        end
    end
    
    
    %social=max(social,passing);%comment out to get rid of the passing layer
    cost=social;
end

function v=classifyCost(c)
    if(c>0.8)
        v=0.8;
    elseif(c>0.6)
        v=0.6;
    elseif(c>0.4)
        v=0.4;
    else
        v=0.2;
    end
end

function v=asymmetricGaus(x,y,x0,y0,theta,sigmaFront,sigmaSide,sigmaBack)

    angle=mod(atan2(y-y0,x-x0),2*pi)-theta;%basically the normalization step from the paper
    
    if (abs(angle)>=pi/2&&abs(angle)<=pi+pi/2)%changed a little since the normalization is slightly different
        sigma=sigmaBack;
    else
        sigma=sigmaFront;
    end

    a=((cos(theta)^2)/(2*sigma^2))  + ((sin(theta)^2)/(2*sigmaSide^2));
    b=(sin(2*theta)/(4*sigma^2))    -  (sin(2*theta)/(4*sigmaSide^2));
    c=((sin(theta)^2)/(2*sigma^2))  + ((cos(theta)^2)/(2*sigmaSide^2));
    
    v=exp(-(a*(x-x0)^2+2*b*(x-x0)*(y-y0)+c*(y-y0)^2));
end 