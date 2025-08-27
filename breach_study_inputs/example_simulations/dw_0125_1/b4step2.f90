! ============================================
subroutine b4step2(mbc,mx,my,meqn,q,xlower,ylower,dx,dy,t,dt,maux,aux)
! ============================================
!
! # called before each call to step
! # use to set time-dependent aux arrays or perform other tasks.
!
! This particular routine sets negative values of q(1,i,j) to zero,
! as well as the corresponding q(m,i,j) for m=1,meqn.
! This is for problems where q(1,i,j) is a depth.
! This should occur only because of rounding error.
!
! Also calls movetopo if topography might be moving.

    use geoclaw_module, only: dry_tolerance
    use geoclaw_module, only: g => grav
    use topo_module, only: num_dtopo,topotime
    use topo_module, only: aux_finalized
    use topo_module, only: xlowdtopo,xhidtopo,ylowdtopo,yhidtopo

    use amr_module, only: xlowdomain => xlower
    use amr_module, only: ylowdomain => ylower
    use amr_module, only: xhidomain => xupper
    use amr_module, only: yhidomain => yupper
    use amr_module, only: xperdom,yperdom,spheredom,NEEDS_TO_BE_SET

    use storm_module, only: set_storm_fields

    use breach_module, only: center
    use breach_module, only: south
    use breach_module, only: north
    use breach_module, only: west
    use breach_module, only: east
    use breach_module, only: sigma
    use breach_module, only: breach_trigger
    use breach_module, only: time_ratio
    use breach_module, only: start_time
    use breach_module, only: end_time
    use breach_module, only: num_breaches
    use breach_module, only: depth
    implicit none

    ! Subroutine arguments
    integer, intent(in) :: meqn
    integer, intent(inout) :: mbc,mx,my,maux
    real(kind=8), intent(inout) :: xlower, ylower, dx, dy, t, dt
    real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)
!    real(kind=8), intent(in) :: mu, sigma, lat0, lat1, lon0, lon1, time, breach_trigger, time_ratio, start_time, end_time


    ! Local storage
    integer :: i,j, num, status
    real(kind=8) :: x,y
    real(kind=8), dimension(3) :: time_array

    ! Check for NaNs in the solution
    call check4nans(meqn,mbc,mx,my,q,t,1)

    ! check for h < 0 and reset to zero
    ! check for h < drytolerance
    ! set hu = hv = 0 in all these cells
    forall(i=1-mbc:mx+mbc, j=1-mbc:my+mbc,q(1,i,j) < dry_tolerance)
        q(1,i,j) = max(q(1,i,j),0.d0)
        q(2:3,i,j) = 0.d0
    end forall

    if (aux_finalized < 2) then
        print *, 'RESETTING AUX'
        ! topo arrays might have been updated by dtopo more recently than
        ! aux arrays were set unless at least 1 step taken on all levels
        aux(1,:,:) = NEEDS_TO_BE_SET ! new system checks this val before setting
        call setaux(mbc,mx,my,xlower,ylower,dx,dy,maux,aux)
    endif

    call setprob()

    ! Breach
    do num=1,num_breaches
        if (breach_trigger(num) == 1) then
            if ((start_time(num) <= t) .and. (end_time(num) >=t)) then
                do j=1-mbc,my+mbc
                    y = ylower + (j-0.5d0) * dy
                    do i=1-mbc,mx+mbc
                        x = xlower + (i-0.5d0) * dx
                        if ((x > west(num)) .and. (x < east(num)) .and. &
                            (y > south(num)) .and. (y < north(num))) then
                            if (aux(1,i,j) >= depth(num)) then
                                aux(1, i, j) = aux(1,i,j) -& 
                                               (sigma * exp(-0.5 * (x - center(num))**2/sigma**2)) * &
                                               (time_ratio(num) )
                            end if
                        end if
                    end do
                end do
            end if
        end if
    end do

    ! Set wind and pressure aux variables for this grid
    call set_storm_fields(maux,mbc,mx,my,xlower,ylower,dx,dy,t,aux)

end subroutine b4step2
